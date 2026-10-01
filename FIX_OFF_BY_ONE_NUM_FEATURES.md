# Fix off-by-one in `num_features` (`select_scattered_features` returns N+1)

Status: documented, NOT fixed (too many pending changes in the worktree).

## Symptom (observed via `tests/backend.rs`)

- `Detector::builder().num_features(63)` produces templates with **64** features.
- `num_features(70)` produces **71** features.
- Failing assertion that proves it (before the test was relaxed to `>=`):
  `native detector did not use all 63 features: left: 64, right: 63`.

## Location

- `src/pyramid.rs::select_scattered_features` (`src/pyramid.rs:425-478`),
  called from `extract_features` (`src/pyramid.rs:363`).

## Root cause

`src/pyramid.rs:459-475`:

```rust
let mut features: Vec<Feature> = vec![first.f];   // len 1, `next` holds 2nd
while features.len() < num_features && !candidates.is_empty() {
    ...
    features.push(next);                            // fills up to len == N
    next = candidates.swap_remove(furthest_idx).0.f;
}
features.push(next);                                // unconditional +1 -> N+1
```

Trace with abundant candidates and `num_features = N`: the loop exits with
`features.len() == N` and a pending `next`, which is then pushed
unconditionally, returning **N+1**. When candidates are starved (fewer than
N), the loop drains the vec and the trailing push completes the set, so the
starved case correctly returns all C candidates.

## Consequences

1. **The u8 accumulator branch is dead with default settings.**
   `match_templates` (`src/line2dup.rs:349-363`) picks the u16 accumulator
   when any template has `features.len() >= 64`. The default
   `num_features = 63` actually stores 64 features, so the u16 branch is
   always taken; the `63` parametrization in `tests/backend.rs` does NOT
   cover the u8 branch (a value like `60` -> 61 features would).
2. **Similarity thresholds scale with the inflated length.**
   `src/line2dup.rs:411,441` compute `threshold * 4.0 * features.len()`,
   so the effective threshold is `(N+1)/N` higher than a user configuring
   exactly N features would expect.
3. **Latent panic on a single candidate.** With exactly one candidate,
   `candidates_iter.next()` consumes it, the collected vec is empty, and
   `candidates.swap_remove(0)` (`src/pyramid.rs:456`) panics. Unrelated to
   the +1 but in the same function; fix together.
4. No C++ parity concern: the C++ `selectScatteredFeatures`
   (`cpp/shape_based_matching/line2Dup.cpp:151-200`) is a
   distance-relaxation loop with no N+1 structure, so correcting the count
   does not diverge from the reference. Note though that existing
   similarity expectations were tuned with N+1 lengths and may shift
   slightly after the fix.

## Fix options (pick one when the tree is clean)

- **A (minimal):** change the loop to
  `while features.len() + 1 < num_features && !candidates.is_empty()`,
  keeping the trailing push — abundant case returns exactly N, starved
  case still returns all C.
- **B:** drop the trailing push and keep the loop as-is — but then the
  starved case loses the last pending `next`; needs an explicit
  `if candidates.is_empty() { features.push(next); }`.
- Either way, guard the single-candidate case (early return the lone
  feature instead of `swap_remove` on an empty vec).

## Verification plan (per `AGENTS.md`: failing test first)

1. Tighten `tests/backend.rs::check_backend_parity` to
   `assert_eq!(len, num_features)` for both 63 and 70 (fails today:
   64/71).
2. Apply the fix.
3. Run `cargo test` and `cargo test --features opencv` (backend parity
   then genuinely covers u8 via 63 and u16 via 70).
