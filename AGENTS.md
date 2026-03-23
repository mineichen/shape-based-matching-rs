# Find multiple occurances of a object from a reference image in a different image based on the shape

## Setup
This is a rust-port of a CPP lib, which is checked out in the 'cpp' folder. It should behave the same way eventually. The rust version is currently much slower, but should catch up over time.

## Performance characteristics
This library has to be highly performant. Priorize low number of allocations over code readability.

We'll eventually get rid of opencv to be able to run it in a browser.

# Environment
You are running in a isolated container. 
If you run tests, always use the --release flag, 
so your builds don't overlap with the one of my editor (causing slow builds).
Configure your rust-analyzer/LSP to use a different target directory (e.g. target-opencode) 
to avoid build cache invalidation conflicts with the IDE's LSP.

# Coordinate System
Use a right-handed coordinate-system with first pixel (0,0) beign in the top-left corner. The point at 0,0 should be in the center of the first pixel.

# Angle Convention
All angles are in degrees, with **positive meaning clockwise** rotation. This differs from the C++ implementation which uses negative angles for CW rotation.
