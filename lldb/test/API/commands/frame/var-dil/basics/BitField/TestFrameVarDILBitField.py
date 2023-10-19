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

class TestFrameVarDILBitField(TestBase):
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

        self.expect("frame variable --dil 'bf.a'", substrs=["1023"])
        self.expect("frame variable --dil 'bf.b'", substrs=["9"])
        self.expect("frame variable --dil 'bf.c'", substrs=["false"])
        self.expect("frame variable --dil 'bf.d'", substrs=["true"])

        # Perform an operation to ensure we actually read the value.
        self.expect("frame variable --dil '0 + bf.a'", substrs=["1023"])
        self.expect("frame variable --dil '0 + bf.b'", substrs=["9"])
        self.expect("frame variable --dil '0 + bf.c'", substrs=["0"])
        self.expect("frame variable --dil '0 + bf.d'", substrs=["1"])

        self.expect("frame variable --dil 'abf.a'", substrs=["1023"])
        self.expect("frame variable --dil 'abf.b'", substrs=["'\\x0f'"])
        self.expect("frame variable --dil 'abf.c'", substrs=["3"])

        # Perform an operation to ensure we actually read the value.
        self.expect("frame variable --dil 'abf.a + 0'", substrs=["1023"])
        self.expect("frame variable --dil 'abf.b + 0'", substrs=["15"])
        self.expect("frame variable --dil 'abf.c + 0'", substrs=["3"])

        # Address-of is not allowed for bit-fields.
        self.expect("frame variable --dil '&bf.a'", error=True,
                    substrs=["address of bit-field requested"])
        self.expect("frame variable --dil '&(true ? bf.a : bf.a)'", error=True,
                    substrs=["address of bit-field requested"])
