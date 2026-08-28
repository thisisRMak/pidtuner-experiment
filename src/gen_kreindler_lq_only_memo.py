#!/usr/bin/env python3
"""Generates docs/memos/2026-08-25/2026-08-25-kreindler-lq-only-memo.{md,html}
— a one-pager proposing how to bring AIKreindlerRothschildModelFollowingN.m
into the preset catalog as a plain LQR ("LQ only") plant, per the
professor's 2026-08-25 email guidance, with the specific open decisions
that need his input before implementing. Static/discussion content, not
computed from a live run — no CLI/library calls needed.

Run: python3 gen_kreindler_lq_only_memo.py
"""
import os

_HERE = os.path.dirname(__file__)
_OUT_DIR = os.path.join(_HERE, "..", "docs", "memos", "2026-08-25")
_CSS = open(os.path.join(_HERE, "_memo_css.txt")).read()
_BASENAME = "2026-08-25-kreindler-lq-only-memo"

BODY_HTML = """
<h1>Kreindler-Rothschild as "LQ Only" &mdash; Proposed Scope</h1>

<dl class="memo-header">
  <dt>Date</dt><dd>2026-08-25</dd>
  <dt>Re</dt><dd>Follow-up to the professor's guidance that
  <code>AIKreindlerRothschildModelFollowingN.m</code> ("no 2.m for this example. *N.m is the
  latest. I would keep it as is.") should be brought into the catalog as an LQ-only design, not
  its full model-following apparatus. Proposed scope below, with the specific decisions that
  need the professor's input before implementing.</dd>
</dl>

<h2>What's in the source file today</h2>
<p>
<code>AIKreindlerRothschildModelFollowingN.m</code> is structurally different from the other 12
catalog plants: instead of one plant + one <code>lqr()</code> call, it builds the F-4 airframe
(<code>Aa</code>, <code>Ba</code>, 4 states/2 inputs), then augments it twice &mdash; once for
implicit model-following, once for explicit &mdash; adding actuator dynamics
(<code>Adelta</code>/<code>Bdelta</code>), a reference model (<code>Aaprime</code>/
<code>Baprime</code>), and a command generator (<code>Ac</code>/<code>Cc</code>), with its own
<code>lqr()</code> call and closed-loop simulation for each augmented system. None of that
matches the catalog's plain-LQR shape, which is why this file was never transcribed into
<code>lqg_examples_json/</code> in the first place.
</p>

<h2>Proposed "LQ only" scope</h2>
<p>
Reading "LQ only" as: keep just the bare airframe as a plain regulator problem, the same shape
as the other 12 presets, and drop the model-following scaffolding entirely.
</p>
<ul>
<li><b>Plant:</b> <code>Aa</code>/<code>Ba</code> as given (4 states, 2 inputs) &mdash; no
actuator augmentation, no reference model, no command generator.</li>
<li><b>Design method:</b> one plain <code>LQR</code> design (matching the catalog's
<code>LQR</code>/<code>OutputWeightedLQR</code>/<code>Bryson</code> pattern), not implicit or
explicit model-following.</li>
<li><b>Catalog slot:</b> a 13th preset in <code>lqg_examples_gen.py</code>/
<code>lqg_examples_json/</code>, still sourced from this original (non-<code>*2</code>) file,
following the same <code>build_*()</code> &rarr; JSON pattern as the other 12.</li>
</ul>

<h2>What needs the professor's input before implementing</h2>
<p>
The source file never defines a plain <code>C</code>, <code>Q</code>, or <code>R</code> for the
bare <code>Aa</code>/<code>Ba</code> airframe outside the model-following context &mdash; every
<code>Q</code>/<code>R</code>/<code>C</code> in the file (<code>Qhat</code>/<code>Rhat</code> for
implicit, <code>Q</code>/<code>R</code>/<code>C</code>=I(4) for explicit) is built for one of the
augmented systems, not the bare plant. This is the same "unspecified output map" situation
Chemical Reactor had, but there's no obvious default to fall back on this time since even the
augmented systems' weights aren't really "the same problem, just augmented" &mdash; they're
weighting different things (model-following error, not the airframe's own states/outputs).
</p>
<div class="callout">
<strong>Questions for the professor:</strong>
<ol>
<li>Is <code>Aa</code>/<code>Ba</code> (the bare airframe, before actuator/model/command-generator
augmentation) the right plant to extract for an "LQ only" treatment, or did you have a different
subset in mind?</li>
<li>What output map <code>C</code> should we use for a plain LQR design on this plant &mdash;
full state feedback (<code>C=I(4)</code>, our usual fallback when nothing is specified), or a
specific measurement set?</li>
<li>What <code>Q</code>/<code>R</code> would you suggest for this plant's plain LQR design?
Nothing in the source file is a direct fit (the existing <code>Q</code>/<code>R</code>/
<code>Qhat</code>/<code>Rhat</code> values all belong to the model-following augmentations,
not a bare-airframe regulator problem) &mdash; is <code>Q=I(4)</code>, <code>R=I(2)</code> (our
identity fallback used elsewhere in the catalog) an acceptable stand-in, or would you rather
specify weights for this plant directly?</li>
</ol>
</div>

<h2>Not proposed</h2>
<p>
The implicit/explicit model-following designs already in the file, and the repo's existing
general-purpose model-following code (<code>lqg_implicit.py</code>, <code>lqg_explicit.py</code>),
are untouched by this proposal &mdash; "LQ only" is specifically about adding a second, simpler
catalog entry alongside that, not replacing or reworking the model-following material.
</p>
"""

BODY_MD = """# Kreindler-Rothschild as "LQ Only" — Proposed Scope

Date: 2026-08-25
Re: Follow-up to the professor's guidance that `AIKreindlerRothschildModelFollowingN.m` ("no
2.m for this example. *N.m is the latest. I would keep it as is.") should be brought into the
catalog as an LQ-only design, not its full model-following apparatus. Proposed scope below, with
the specific decisions that need the professor's input before implementing.

---

## What's in the source file today

`AIKreindlerRothschildModelFollowingN.m` is structurally different from the other 12 catalog
plants: instead of one plant + one `lqr()` call, it builds the F-4 airframe (`Aa`, `Ba`, 4
states/2 inputs), then augments it twice — once for implicit model-following, once for explicit
— adding actuator dynamics (`Adelta`/`Bdelta`), a reference model (`Aaprime`/`Baprime`), and a
command generator (`Ac`/`Cc`), with its own `lqr()` call and closed-loop simulation for each
augmented system. None of that matches the catalog's plain-LQR shape, which is why this file was
never transcribed into `lqg_examples_json/` in the first place.

## Proposed "LQ only" scope

Reading "LQ only" as: keep just the bare airframe as a plain regulator problem, the same shape as
the other 12 presets, and drop the model-following scaffolding entirely.

- **Plant:** `Aa`/`Ba` as given (4 states, 2 inputs) — no actuator augmentation, no reference
  model, no command generator.
- **Design method:** one plain `LQR` design (matching the catalog's
  `LQR`/`OutputWeightedLQR`/`Bryson` pattern), not implicit or explicit model-following.
- **Catalog slot:** a 13th preset in `lqg_examples_gen.py`/`lqg_examples_json/`, still sourced
  from this original (non-`*2`) file, following the same `build_*()` → JSON pattern as the other
  12.

## What needs the professor's input before implementing

The source file never defines a plain `C`, `Q`, or `R` for the bare `Aa`/`Ba` airframe outside
the model-following context — every `Q`/`R`/`C` in the file (`Qhat`/`Rhat` for implicit,
`Q`/`R`/`C`=I(4) for explicit) is built for one of the augmented systems, not the bare plant.
This is the same "unspecified output map" situation Chemical Reactor had, but there's no obvious
default to fall back on this time since even the augmented systems' weights aren't really "the
same problem, just augmented" — they're weighting different things (model-following error, not
the airframe's own states/outputs).

**Questions for the professor:**

1. Is `Aa`/`Ba` (the bare airframe, before actuator/model/command-generator augmentation) the
   right plant to extract for an "LQ only" treatment, or did you have a different subset in mind?
2. What output map `C` should we use for a plain LQR design on this plant — full state feedback
   (`C=I(4)`, our usual fallback when nothing is specified), or a specific measurement set?
3. What `Q`/`R` would you suggest for this plant's plain LQR design? Nothing in the source file
   is a direct fit (the existing `Q`/`R`/`Qhat`/`Rhat` values all belong to the model-following
   augmentations, not a bare-airframe regulator problem) — is `Q=I(4)`, `R=I(2)` (our identity
   fallback used elsewhere in the catalog) an acceptable stand-in, or would you rather specify
   weights for this plant directly?

## Not proposed

The implicit/explicit model-following designs already in the file, and the repo's existing
general-purpose model-following code (`lqg_implicit.py`, `lqg_explicit.py`), are untouched by
this proposal — "LQ only" is specifically about adding a second, simpler catalog entry alongside
that, not replacing or reworking the model-following material.
"""


def wrap_html(title, body):
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
{_CSS}
</style>
</head>
<body>
{body}
</body>
</html>
"""


if __name__ == "__main__":
    os.makedirs(_OUT_DIR, exist_ok=True)
    html_path = os.path.join(_OUT_DIR, f"{_BASENAME}.html")
    with open(html_path, "w") as f:
        f.write(wrap_html("Kreindler-Rothschild LQ-Only Scope", BODY_HTML))
    print(f"wrote {html_path}")
    md_path = os.path.join(_OUT_DIR, f"{_BASENAME}.md")
    with open(md_path, "w") as f:
        f.write(BODY_MD)
    print(f"wrote {md_path}")
