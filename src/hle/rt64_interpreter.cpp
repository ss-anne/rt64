//
// RT64
//

#include "rt64_interpreter.h"

#include <atomic>
#include <cassert>
#include <cstdint>

//#define DUMP_DISPLAY_LISTS

// PSR walk-time GDL CALL-target snapshot probe (paired with the submit-time
// snapshot in extras.c / RT64Context::send_dl). See extras.c for details.
extern "C" void pkmnstadium_gdl_walk_snapshot(
    uint32_t target_vaddr, uint32_t parent_vaddr,
    const uint8_t* head_ptr, uint64_t submit_seq);

namespace RT64 {
    // ── Interpreter probe ring ──────────────────────────────────────
    // Always-on, tracks the last N (host) DL pointers visited in
    // processDisplayLists so a freeze can be diagnosed by reading
    // back the recent loop. The recompiler-side runtime can copy this
    // ring out via interpreter_recent_pcs_copy() at post-mortem time.
    namespace interp_probe {
        constexpr size_t RING_CAP = 16384;
        static std::atomic<uint64_t> g_seq{0};
        static uint64_t g_ring_pc[RING_CAP];   // 64-bit host pointer values
        static std::atomic<uint64_t> g_step{0}; // iterations of inner loop since last task reset
        static std::atomic<uint64_t> g_task_index{0}; // number of processDisplayLists calls so far
        static std::atomic<uint64_t> g_current_pc{0};
        static std::atomic<uint64_t> g_dl_start{0};

        // Control-flow-only ring (G_DL / G_ENDDL). Captures the last
        // ~16K control-flow events; an OOB walk through G_NOOPs blows
        // away the regular ring quickly but doesn't generate CF events,
        // so this preserves the events leading up to the divergence.
        struct CfEvent {
            uint64_t step;        // step counter at the event
            uint64_t pc;          // host pointer of the cmd
            uint32_t w0;          // command word 0 (opcode in bits 24-31)
            uint32_t w1;          // command word 1
            uint64_t target_or_pop; // resolved host target for G_DL, popped value for G_ENDDL
            uint32_t stack_depth_after;
            uint32_t pad;
        };
        constexpr size_t CF_RING_CAP = 16384;
        static std::atomic<uint64_t> g_cf_seq{0};
        static CfEvent g_cf_ring[CF_RING_CAP];

        inline void record_cf(uint64_t pc, uint32_t w0, uint32_t w1,
                              uint64_t target_or_pop, uint32_t depth_after) {
            const uint64_t s = g_cf_seq.fetch_add(1, std::memory_order_relaxed);
            CfEvent& e = g_cf_ring[s % CF_RING_CAP];
            e.step = g_step.load(std::memory_order_relaxed);
            e.pc = pc;
            e.w0 = w0;
            e.w1 = w1;
            e.target_or_pop = target_or_pop;
            e.stack_depth_after = depth_after;
        }
    }

    static FILE *displayListFp = nullptr;

    // Interpreter

    Interpreter::Interpreter() {
        state = nullptr;
        hleGBI = nullptr;
        extendedFunction = gbiManager.getExtendedFunction();
    }

    void Interpreter::setup(State *state) {
        this->state = state;
    }

    void Interpreter::loadUCodeGBI(uint32_t textAddress, uint32_t dataAddress, bool resetFromTask) {
        if (!resetFromTask) {
            state->flush();
        }

        const uint32_t AddressMask = 0xFFFFF8;
        const uint32_t maskedTextAddress = textAddress & AddressMask;
        const uint32_t maskedDataAddress = dataAddress & AddressMask;
        if ((UCode.textAddress != maskedTextAddress) || (UCode.dataAddress != maskedDataAddress)) {
            hleGBI = gbiManager.getGBIForUCode(state->RDRAM, maskedTextAddress, maskedDataAddress);
            if (hleGBI != nullptr) {
                state->rsp->setGBI(hleGBI);
            }

            UCode.textAddress = maskedTextAddress;
            UCode.dataAddress = maskedDataAddress;
        }

        if (hleGBI != nullptr) {
            GBIReset resetFunction = resetFromTask ? hleGBI->resetFromTask : hleGBI->resetFromLoad;
            if (resetFunction != nullptr) {
                resetFunction(state);
            }
        }
    }

    void Interpreter::processRDPLists(uint32_t dlStartAdddress, DisplayList *dlStart, DisplayList *dlEnd) {
        state->dlCpuProfiler.start();

        // Update the state with the current display list address.
        state->displayListAddress = dlStartAdddress;
        state->displayListCounter++;

        // Check RDRAM if required.
        state->checkRDRAM();

        GBI *rdpGBI = state->rdp->gbi;
        constexpr unsigned int opCodeMask = 0x3F;

        // Run the command interpreter.
        assert(rdpGBI != nullptr);
        DisplayList *dl = dlStart;
        uint8_t opCode;
        GBIFunction func;
        uint32_t cmdLength;
        size_t pendingCommandRemainingBytes = state->rdp->pendingCommandRemainingBytes;

        if (dlStart >= dlEnd) {
            state->dlCpuProfiler.end();
            return;
        }

        if (pendingCommandRemainingBytes != 0) {
            // Copy the remaining command bytes from the current displaylist
            uint32_t toCopy = (uint32_t)std::min(pendingCommandRemainingBytes, (uintptr_t)dlEnd - (uintptr_t)dl);
            memcpy(state->rdp->pendingCommandBuffer.data() + state->rdp->pendingCommandCurrentBytes, dl, toCopy);

            // Modify start to skip the copied bytes
            dl = (DisplayList *)(toCopy + (uintptr_t)dl);

            // Check if we've copied all of the bytes of the command into the buffer
            if (pendingCommandRemainingBytes == toCopy) {
                // All bytes have been copied, so run the completed command
                DisplayList *pendingCommand = (DisplayList *)state->rdp->pendingCommandBuffer.data();
                opCode = (pendingCommand->w0 >> 24) & opCodeMask;
                func = rdpGBI->map[opCode];

                if (func != nullptr) {
                    func(state, &pendingCommand);
                }
                else {
                    RT64_LOG_PRINTF("DL Parser ran into an unknown RDP opCode: %u / 0x%X", opCode, opCode);
                }

                state->rdp->pendingCommandCurrentBytes = 0;
                state->rdp->pendingCommandRemainingBytes = 0;
            }
            // Not all of the bytes were copied, so adjust RDP state accordingly and exit.
            else {
                state->rdp->pendingCommandCurrentBytes += toCopy;
                state->rdp->pendingCommandRemainingBytes -= toCopy;
                state->dlCpuProfiler.end();
                return;
            }
        }

        // Create a dummy pointer and pass that, since displaylist pointer incrementing is handled differently in LLE.
        DisplayList *dummy;
        while ((dl != nullptr) && ((dlEnd == nullptr) || (dl < dlEnd))) {
            opCode = (dl->w0 >> 24) & opCodeMask;

            if ((extendedOpCode != 0) && (opCode == extendedOpCode)) {
                dummy = dl;
                extendedFunction(state, &dl);
                cmdLength = 1;
            }
            else {
                func = rdpGBI->map[opCode];
                cmdLength = state->rdp->commandWordLengths[opCode];

#       ifdef DUMP_DISPLAY_LISTS
                RT64_LOG_PRINTF("0x%08X 0x%08X", dl->w0, dl->w1);
#       endif

                // Check if this command is unfinished and store the partial contents if so.
                if (dl + cmdLength > dlEnd) {
                    uint32_t toCopy = (uint32_t)((uintptr_t)dlEnd - (uintptr_t)dl);
                    memcpy(state->rdp->pendingCommandBuffer.data(), dl, toCopy);
                    state->rdp->pendingCommandCurrentBytes = toCopy;
                    state->rdp->pendingCommandRemainingBytes = cmdLength * sizeof(DisplayList) - toCopy;
                    break;
                }

                if (func != nullptr) {
                    dummy = dl;
                    func(state, &dummy);
                }
                else {
                    RT64_LOG_PRINTF("DL Parser ran into an unknown RDP opCode: %u / 0x%X", opCode, opCode);
                }
            }

            if (dl != nullptr) {
                dl += cmdLength;
            }
        }

        state->dlCpuProfiler.end();
    }

    void Interpreter::processDisplayLists(uint32_t dlStartAdddress, DisplayList *dlStart) {
        assert(hleGBI != nullptr);

        state->dlCpuProfiler.start();

        // Update the state with the current display list address.
        state->displayListAddress = dlStartAdddress;
        state->displayListCounter++;

        // Check RDRAM if required.
        state->checkRDRAM();

        // Probe: reset the per-task step counter and remember the start address
        interp_probe::g_step.store(0, std::memory_order_relaxed);
        interp_probe::g_task_index.fetch_add(1, std::memory_order_relaxed);
        interp_probe::g_dl_start.store(dlStartAdddress, std::memory_order_relaxed);

        // Run the command interpreter.
        DisplayList *dl = dlStart;
        uint8_t opCode;
        GBIFunction func;
        while (dl != nullptr) {
            // Probe: record this DL's pointer in the always-on ring
            const uint64_t s = interp_probe::g_seq.fetch_add(1, std::memory_order_relaxed);
            interp_probe::g_ring_pc[s % interp_probe::RING_CAP] =
                reinterpret_cast<uint64_t>(dl);
            interp_probe::g_current_pc.store(reinterpret_cast<uint64_t>(dl),
                                             std::memory_order_relaxed);
            interp_probe::g_step.fetch_add(1, std::memory_order_relaxed);

            // Snapshot for CF probe (cmd may modify dl during dispatch)
            const uint64_t pre_pc = reinterpret_cast<uint64_t>(dl);
            const uint32_t pre_w0 = dl->w0;
            const uint32_t pre_w1 = dl->w1;

            opCode = (dl->w0 >> 24);

            // GDL walk-time probe: when the interpreter is about to walk a
            // sub-DL via G_DL push=1 (CALL), snapshot the target's first 16
            // bytes from RDRAM AS THE INTERPRETER SEES THEM. Pairs with the
            // submit-time snapshot taken in RT64Context::send_dl. Comparing
            // submit-bytes vs walk-bytes for matching target_vaddr identifies:
            //   bytes equal + look like DL    -> normal CALL
            //   bytes equal + don't look like DL -> Stadium emitted G_DL into
            //                                       a non-DL buffer
            //   bytes differ                  -> Stadium overwrote target
            //                                    between submit and walk (race)
            if (opCode == 0xDE && ((pre_w0 >> 16) & 0xFF) == 1) {
                const uint32_t tgt_vaddr = pre_w1;
                const uint32_t tgt_off = tgt_vaddr & 0x7FFFFF;
                if (tgt_off + 16 <= 0x800000 && state->RDRAM != nullptr) {
                    const uint64_t submit_seq =
                        interp_probe::g_task_index.load(std::memory_order_relaxed);
                    /* parent_vaddr: cmd's own kseg0 vaddr. dl host ptr - state->RDRAM
                     * + 0x80000000. Cap to 0x80000000+0x7FFFFF to avoid garbage if
                     * the dl host ptr is from a different mapping. */
                    const uint64_t pc_off =
                        reinterpret_cast<uint64_t>(dl) - reinterpret_cast<uint64_t>(state->RDRAM);
                    const uint32_t parent_vaddr = (pc_off < 0x800000)
                        ? (0x80000000u | static_cast<uint32_t>(pc_off))
                        : 0;
                    pkmnstadium_gdl_walk_snapshot(
                        tgt_vaddr, parent_vaddr,
                        state->RDRAM + tgt_off, submit_seq);
                }
            }

            if ((extendedOpCode != 0) && (opCode == extendedOpCode)) {
                extendedFunction(state, &dl);
            }
            else {
                func = hleGBI->map[opCode];

#       ifdef DUMP_DISPLAY_LISTS
                RT64_LOG_PRINTF("0x%08X 0x%08X", dl->w0, dl->w1);
#       endif

                if (func != nullptr) {
                    func(state, &dl);
                }
                else {
                    RT64_LOG_PRINTF("DL Parser ran into an unknown opCode (GBI %u): %u / 0x%X", uint32_t(hleGBI->ucode), opCode, opCode);
                }
            }

            // Record control-flow events (G_DL=0xDE, G_ENDDL=0xDF). The
            // post-dispatch dl is "target-1" for G_DL or popped value for
            // G_ENDDL (the loop's `dl++` below adjusts G_DL but not the
            // popped value, since popReturnAddress stored pc+1).
            if (opCode == 0xDE || opCode == 0xDF) {
                const uint64_t post_dl = reinterpret_cast<uint64_t>(dl);
                interp_probe::record_cf(pre_pc, pre_w0, pre_w1, post_dl, 0);
            }

            if (dl != nullptr) {
                dl++;
            }
        }

        state->dlCpuProfiler.end();
    }
};

// ── Interpreter probe accessors (extern "C" for the runtime to read) ──
extern "C" uint64_t rt64_interp_seq(void) {
    return RT64::interp_probe::g_seq.load(std::memory_order_relaxed);
}
extern "C" uint64_t rt64_interp_step(void) {
    return RT64::interp_probe::g_step.load(std::memory_order_relaxed);
}
extern "C" uint64_t rt64_interp_task_index(void) {
    return RT64::interp_probe::g_task_index.load(std::memory_order_relaxed);
}
extern "C" uint64_t rt64_interp_current_pc(void) {
    return RT64::interp_probe::g_current_pc.load(std::memory_order_relaxed);
}
extern "C" uint64_t rt64_interp_dl_start(void) {
    return RT64::interp_probe::g_dl_start.load(std::memory_order_relaxed);
}
// Copy up to `cap` of the most recent ring entries (host pointers) into
// `out`. Returns the number written and the seq value at copy-time.
extern "C" void rt64_interp_recent_copy(uint64_t* out, size_t cap,
                                         size_t* n_written, uint64_t* seq_out) {
    using namespace RT64::interp_probe;
    const uint64_t s = g_seq.load(std::memory_order_relaxed);
    if (seq_out) *seq_out = s;
    if (cap == 0 || out == nullptr) {
        if (n_written) *n_written = 0;
        return;
    }
    const size_t available = (s < RING_CAP) ? size_t(s) : RING_CAP;
    const size_t want = (cap < available) ? cap : available;
    const size_t start = (s - want) % RING_CAP;
    for (size_t i = 0; i < want; i++) {
        out[i] = g_ring_pc[(start + i) % RING_CAP];
    }
    if (n_written) *n_written = want;
}

extern "C" size_t rt64_interp_cf_event_size(void) {
    return sizeof(RT64::interp_probe::CfEvent);
}
extern "C" void rt64_interp_cf_recent_copy(void* out_void, size_t cap,
                                           size_t* n_written,
                                           uint64_t* seq_out) {
    using namespace RT64::interp_probe;
    const uint64_t s = g_cf_seq.load(std::memory_order_relaxed);
    if (seq_out) *seq_out = s;
    if (cap == 0 || out_void == nullptr) {
        if (n_written) *n_written = 0;
        return;
    }
    const size_t available = (s < CF_RING_CAP) ? size_t(s) : CF_RING_CAP;
    const size_t want = (cap < available) ? cap : available;
    const size_t start = (s - want) % CF_RING_CAP;
    CfEvent* out = static_cast<CfEvent*>(out_void);
    for (size_t i = 0; i < want; i++) {
        out[i] = g_cf_ring[(start + i) % CF_RING_CAP];
    }
    if (n_written) *n_written = want;
}
