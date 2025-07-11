## File description

- The file `graphics.py` is a simple pygame representation of the problem of calculating a robot's position using beacons
- `particle_filter.c` is a C implementation of a particle filter that uses RSSI measurements to calculate the robot's position

## Notes
- `graphics.py` doesn't use `particle_filter.c`
- `graphics.py` is mean't for conceptual analysis
- The main() function in `particle_filter.c` is a usage example
- The particle filter implementation in `particle_filter.c` is meant for embedded

## TO-DO
- Test and benchmark the particle filter implementation on a nRF5340's app core
- Properly document usage of the implementation
