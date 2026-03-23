# Find multiple occurances of a object from a reference image in a different image based on the shape

## Setup
This is a rust-port of a CPP lib, which is checked out in the 'cpp' folder. It should behave the same way eventually. The rust version is currently much slower, but should catch up over time.

## Performance characteristics
This library has to be highly performant. Priorize low number of allocations over code readability.

We'll eventually get rid of opencv to be able to run it in a browser.

# Environment
You are running in a isolated container. 

# Definition of done
**IMPORTANT** Always write a failing automated test before you write new production code to avoid regressions in the future.
After changeing production code, always run the test suite to catch potential regressions.
You only return or start removing print statements, when all tests run successfully. 

# Coordinate System
Use a right-handed coordinate-system with first pixel (0,0) beign in the top-left corner. The point at 0,0 should be in the center of the first pixel.

# Algorithm properties
- All angles are in degrees, with **positive meaning clockwise** rotation. This differs from the C++ implementation which uses negative angles for CW rotation.
- If matching templates are rotated or scaled around a origin
