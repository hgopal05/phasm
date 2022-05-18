
#include <stdio.h>
#include <map>
#include <iostream>
#include "pin.H"
#include "utils.hpp"

#include "interpreter.cpp"
// This is a hack because PIN has its own complicated Makefiles that I don't fully understand

// FILE* trace;
bool target_function_found = false;
Interpreter* interpreter = nullptr;
std::vector<std::string> routine_names;
int current_routine = -1;
int target_routine = -1;

KNOB< std::string > KnobTargetFunction(KNOB_MODE_WRITEONCE, "pintool", "f", "target6(int)", "Specify name of function to target");

VOID record_read_ins(VOID* ip, VOID* addr, UINT32 len, ADDRINT rbp, ADDRINT rsp) {
    interpreter->read_mem(ip, addr, len, rbp, rsp);
}

VOID record_write_ins(VOID* ip, VOID* addr, UINT32 len, ADDRINT rbp, ADDRINT rsp) {
    interpreter->write_mem(ip, addr, len, rbp, rsp);
}

VOID record_enter_target_rtn(UINT64 routine_id, VOID* ip, ADDRINT rsp) {
    interpreter->enter_fun(ip, routine_id, rsp);  // rsp turns into rbp after prolog completes
}

VOID record_exit_target_rtn(UINT64 routine_id, VOID* ip) {
    interpreter->exit_fun(ip);
}

VOID record_enter_rtn(UINT64 routine_id, VOID* ip, ADDRINT rsp) {
    interpreter->enter_fun(ip, routine_id, rsp);
}

VOID record_exit_rtn(UINT64 routine_id, VOID* ip) {
    interpreter->exit_fun(ip);
}

VOID record_malloc_first_argument(ADDRINT size, VOID* ip) {
    interpreter->request_malloc(ip, size);
}

VOID record_malloc_return(ADDRINT addr, VOID* ip) {
    interpreter->receive_malloc(ip, addr);
}

VOID record_free_first_argument(ADDRINT addr, VOID* ip) {
    interpreter->free(ip, addr);
};

VOID instrument_ins(INS ins, VOID* v) {
    // Instruments memory accesses using a predicated call, i.e.
    // the instrumentation is called iff the instruction will actually be executed.

    // On the IA-32 and Intel(R) 64 architectures conditional moves and REP
    // prefixed instructions appear as predicated instructions in Pin.
    UINT32 memOperands = INS_MemoryOperandCount(ins);

    // Iterate over each memory operand of the instruction.
    for (UINT32 memOp = 0; memOp < memOperands; memOp++)
    {
        if (INS_MemoryOperandIsRead(ins, memOp)) {
            INS_InsertPredicatedCall(ins, IPOINT_BEFORE, (AFUNPTR) record_read_ins,
                                     IARG_INST_PTR,
                                     IARG_MEMORYOP_EA, memOp,
                                     IARG_MEMORYOP_SIZE, memOp,
                                     IARG_REG_VALUE, REG_RBP,
                                     IARG_REG_VALUE, REG_RSP,
                                     IARG_END);
        }

        // Note that in some architectures a single memory operand can be
        // both read and written (for instance incl (%eax) on IA-32)
        // In that case we instrument it once for read and once for write.
        if (INS_MemoryOperandIsWritten(ins, memOp)) {
            INS_InsertPredicatedCall(ins, IPOINT_BEFORE, (AFUNPTR) record_write_ins,
                                     IARG_INST_PTR,
                                     IARG_MEMORYOP_EA, memOp,
                                     IARG_MEMORYOP_SIZE, memOp,
                                     IARG_REG_VALUE, REG_RBP,
                                     IARG_REG_VALUE, REG_RSP, // We'll want this later
                                     IARG_END);
        }
    }
}

// Called every time a new routine is jitted
void instrument_rtn(RTN rtn, VOID* v) {

    std::string rtn_name = demangle(RTN_Name(rtn));
    ADDRINT rtn_address = RTN_Address(rtn);

    routine_names.push_back(rtn_name);
    current_routine++;

    // Instrument target routine to set in_target_routine to be true
    if (rtn_name == KnobTargetFunction.Value()) {
        printf("Instrumenting %s (%llu)\n", rtn_name.c_str(), current_routine);
        target_function_found = true;
        target_routine = current_routine;

        RTN_Open(rtn);
        // Insert a call to record_enter_rtn at the routine's entry point
        RTN_InsertCall(rtn, IPOINT_BEFORE, (AFUNPTR) record_enter_target_rtn,
                       IARG_UINT64, current_routine,
                       IARG_ADDRINT, rtn_address,
                       IARG_REG_VALUE, REG_RSP,
                       IARG_END);

        // Insert a call to record_exit_rtn at the routine's exit point
        // (Warning: PIN might not find all exit points!)
        RTN_InsertCall(rtn, IPOINT_AFTER, (AFUNPTR) record_exit_target_rtn,
                       IARG_UINT64, current_routine,
                       IARG_ADDRINT, rtn_address,
                       IARG_END);
        RTN_Close(rtn);
    }
    else if (rtn_name == "malloc" || rtn_name == "_malloc") {
        RTN_Open(rtn);
        RTN_InsertCall(rtn, IPOINT_BEFORE, (AFUNPTR) record_malloc_first_argument, IARG_FUNCARG_ENTRYPOINT_VALUE, 0, IARG_INST_PTR, IARG_END);
        RTN_InsertCall(rtn, IPOINT_AFTER, (AFUNPTR) record_malloc_return, IARG_FUNCRET_EXITPOINT_VALUE, IARG_INST_PTR, IARG_END);
        RTN_Close(rtn);
    }
    else if (rtn_name == "free" || rtn_name == "_free") {
        RTN_Open(rtn);
        RTN_InsertCall(rtn, IPOINT_BEFORE, (AFUNPTR) record_free_first_argument, IARG_FUNCARG_ENTRYPOINT_VALUE, 0, IARG_INST_PTR,IARG_END);
        RTN_Close(rtn);
    }
    else {
        RTN_Open(rtn);
        RTN_InsertCall(rtn, IPOINT_BEFORE, (AFUNPTR) record_enter_rtn,
                       IARG_UINT64, current_routine,
                       IARG_ADDRINT, rtn_address,
                       IARG_REG_VALUE, REG_RSP,
                       IARG_END);

        RTN_InsertCall(rtn, IPOINT_AFTER, (AFUNPTR) record_exit_rtn,
                       IARG_UINT64, current_routine,
                       IARG_ADDRINT, rtn_address,
                       IARG_END);
        RTN_Close(rtn);
    }
}

VOID instrument_finish(INT32 code, VOID* v) {
    // printf("#eof\n");
    // fclose(trace);
    if (!target_function_found) {
        std::cout << "Couldn't find target function. Are you sure your binary has debug symbols?" << std::endl;
    }
    else {
        interpreter->print_variables(std::cout);
    }
}


INT32 print_usage() {
    PIN_ERROR("This Pintool prints a trace of memory addresses\n" + KNOB_BASE::StringKnobSummary() + "\n");
    return -1;
}

int main(int argc, char* argv[]) {

    for (int i=0; i<argc; ++i) {
        std::cout << argv[i] << std::endl;
    }

    PIN_InitSymbols();
    if (PIN_Init(argc, argv)) return print_usage();

    std::cout << "Targeting function " << KnobTargetFunction.Value() << std::endl;

    // trace = fopen("pinatrace.out", "w");
    RTN_AddInstrumentFunction(instrument_rtn, 0);
    INS_AddInstrumentFunction(instrument_ins, 0);
    PIN_AddFiniFunction(instrument_finish, 0);

    interpreter = new Interpreter(target_routine, routine_names);
    PIN_StartProgram(); // Never returns
    return 0;
}

