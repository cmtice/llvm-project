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

class TestFrameVarDILValueScope(TestBase):
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

#        self.expect("frame variable --dil a", substrs=["1"])

#  EXPECT_THAT(Scope("var").Eval("x_"), IsEqual("1"));
           # blah = frame.FindVariable("var")
           # EvaluatorHelper helper(blah, compare_with_lldb_, allow_side_effects_)

#  EXPECT_THAT(Scope("var").Eval("y_"), IsEqual("2.5"));
#  EXPECT_THAT(Scope("var").Eval("z_"),
#              IsError("use of undeclared identifier 'z_'"));

#  // In "value" scope `this` refers to the scope object.
#  EXPECT_THAT(Scope("var").Eval("this->y_"), IsEqual("2.5"));
#  EXPECT_THAT(Scope("var").Eval("(*this).y_"), IsEqual("2.5"));

#  // Test for the "artificial" value, i.e. created by the expression.
#  lldb::SBError error;
#  lldb::SBValue scope_var =
#      lldb_eval::EvaluateExpression(frame_, "(test_scope::Value&)bytes", error);
#  EXPECT_TRUE(scope_var.IsValid());
#  EXPECT_TRUE(error.Success());

#  EvaluatorHelper scope(scope_var, true, false);
#  EXPECT_THAT(scope.Eval("this->y_"), IsEqual("2.5"));
#  EXPECT_THAT(scope.Eval("(*this).y_"), IsEqual("2.5"));

#  EXPECT_THAT(Eval("x_"), IsError("use of undeclared identifier 'x_'"));
#  EXPECT_THAT(Eval("y_"), IsError("use of undeclared identifier 'y_'"));
#  EXPECT_THAT(Eval("z_"), IsEqual("3"));

#  // In the frame context `this` is not available here.
#  EXPECT_THAT(
#      Eval("this->y_"),
#      IsError("invalid use of 'this' outside of a non-static member function"));
#  EXPECT_THAT(
#      Eval("(*this)->y_"),
#      IsError("invalid use of 'this' outside of a non-static member function"));

#  EXPECT_THAT(Scope("var").Eval("this - (test_scope::Value*)this"),
#              IsEqual("0"));
