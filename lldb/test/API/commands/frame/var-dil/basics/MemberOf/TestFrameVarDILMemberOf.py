"""
Make sure 'frame var' using DIL parser/evaultor works for local variables.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test import lldbutil

import os
import shutil
import time

class TestFrameVarDILMemberOf(TestBase):
    # If your test case doesn't stress debug info, then
    # set this to true.  That way it won't be run once for
    # each debug info format.
    NO_DEBUG_INFO_TESTCASE = True

    def test_frame_var(self):
        self.build()
        self.do_test()

    def do_test(self):
        target = self.createTestTarget()

        # Now create a breakpoint in main.c at the source matching
        # "Set a breakpoint here"
        breakpoint = target.BreakpointCreateBySourceRegex(
            "Set a breakpoint here", lldb.SBFileSpec("main.cpp")
        )
        self.assertTrue(
            breakpoint and breakpoint.GetNumLocations() >= 1, VALID_BREAKPOINT
        )

        error = lldb.SBError()
        # This is the launch info.  If you want to launch with arguments or
        # environment variables, add them using SetArguments or
        # SetEnvironmentEntries

        launch_info = target.GetLaunchInfo()
        process = target.Launch(launch_info, error)
        self.assertTrue(process, PROCESS_IS_VALID)

        # Did we hit our breakpoint?
        from lldbsuite.test.lldbutil import get_threads_stopped_at_breakpoint

        threads = get_threads_stopped_at_breakpoint(process, breakpoint)
        self.assertEqual(
            len(threads), 1, "There should be a thread stopped at our breakpoint"
        )
       # The hit count for the breakpoint should be 1.
        self.assertEquals(breakpoint.GetHitCount(), 1)

        frame = threads[0].GetFrameAtIndex(0)
        command_result = lldb.SBCommandReturnObject()
        interp = self.dbg.GetCommandInterpreter()

        self.expect("frame variable --dil 's.x'", substrs=["1"])
        self.expect("frame variable --dil 's.r'", substrs=["2"])
        self.expect("frame variable --dil 's.r + 1'", substrs=["3"])
        self.expect("frame variable --dil 'sr.x'", substrs=["1"])
        self.expect("frame variable --dil 'sr.r'", substrs=["2"])
        self.expect("frame variable --dil 'sr.r + 1'", substrs=["3"])
        self.expect("frame variable --dil 'sp->x'", substrs=["1"])
        self.expect("frame variable --dil 'sp->r'", substrs=["2"])
        self.expect("frame variable --dil 'sp->r + 1'", substrs=["3"])
        self.expect("frame variable --dil 'sarr->x'", substrs=["5"]);
        self.expect("frame variable --dil 'sarr->r'", substrs=["2"])
        self.expect("frame variable --dil 'sarr->r + 1'", substrs=["3"])
        self.expect("frame variable --dil '(sarr + 1)->x'", substrs=["1"])

        self.expect("frame variable --dil 'sp->4'", error=True,
                    substrs=["<expr>:1:5: expected 'identifier', got: <'4' "
                             "(numeric_constant)>\n"
                             "sp->4\n"
                             "    ^"])
        self.expect("frame variable --dil 'sp->foo'", error=True,
                    substrs=["no member named 'foo' in 'Sx'"])
        self.expect("frame variable --dil 'sp->r / (void*)0'", error=True,
                    substrs=["invalid operands to binary expression ('int' and "
                             "'void *')"])

        self.expect("frame variable --dil 'sp.x'", error=True,
                    substrs=["member reference type 'Sx *' is a "
                             "pointer; did you mean to use '->'"])
        self.expect("frame variable --dil 'sarr.x'", error=True,
                    substrs=["member reference base type 'Sx[2]' is not a "
                             "structure or union"])

        # Test for record typedefs.
        self.expect("frame variable --dil 'sa.x'", substrs=["3"])
        self.expect("frame variable --dil 'sa.y'", substrs=["'\\x04'"])
