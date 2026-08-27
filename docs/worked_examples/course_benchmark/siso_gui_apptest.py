import sys
sys.path.insert(0, "../../src")
from streamlit.testing.v1 import AppTest

at = AppTest.from_file(
    "../../../src/streamlit_app.py",
    default_timeout=60,
)
at.run()
assert not at.exception, at.exception

# Select the SISO tab's ZN-II method and set L, then click Tune & simulate.
at.selectbox(key="siso_method").set_value("3. Ziegler–Nichols II (ultimate gain)")
at.number_input(key="siso_L").set_value(0.5)
at.run()
assert not at.exception, at.exception

at.button(key="siso_tune").click()
at.run()
assert not at.exception, [str(e) for e in at.exception]

print("=== markdown/text blocks ===")
for md in at.markdown:
    print(md.value)
for t in at.text:
    print(t.value)
for cap in at.caption:
    print(cap.value)

print("=== session state N default check ===")
print("N:", at.session_state["N"])
import inspect
import pid_simulate
print("simulate_closed_loop default N:", inspect.signature(pid_simulate.simulate_closed_loop).parameters["N"].default)


print("OK - GUI (headless AppTest) run completed with no exceptions.")
