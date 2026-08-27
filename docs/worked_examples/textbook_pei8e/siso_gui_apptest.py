import sys
sys.path.insert(0, "../../src")
from streamlit.testing.v1 import AppTest

at = AppTest.from_file(
    "../../../src/streamlit_app.py",
    default_timeout=60,
)
at.run()
assert not at.exception, at.exception

# Switch the SISO tab's plant to the PEI8e textbook surrogate (default is
# the course benchmark plant), then tune ZN-I and ZN-II in turn on it.
at.text_input(key="siso_tf_expr").set_value("1/(90s+1)")
at.number_input(key="siso_L").set_value(13)
at.run()
assert not at.exception, at.exception

for label, method in (
    ("ZN-I",  "2. Ziegler–Nichols I (step / FOPDT)"),
    ("ZN-II", "3. Ziegler–Nichols II (ultimate gain)"),
):
    at.selectbox(key="siso_method").set_value(method)
    at.run()
    assert not at.exception, at.exception

    at.button(key="siso_tune").click()
    at.run()
    assert not at.exception, [str(e) for e in at.exception]

    print(f"=== {label}: info block ===")
    for info in at.info:
        print(info.value)

print("=== session state N default check ===")
print("N:", at.session_state["N"])
import inspect
import pid_simulate
print("simulate_closed_loop default N:", inspect.signature(pid_simulate.simulate_closed_loop).parameters["N"].default)

print("OK - GUI (headless AppTest) run completed with no exceptions.")
