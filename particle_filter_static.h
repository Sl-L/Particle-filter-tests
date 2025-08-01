typedef struct {
    float x;
    float y;
} Point;

typedef struct {
    Point pos;
    float weight;
} Particle;

typedef struct {
    Point min;
    Point max;
} Area;

float rand_uniform(float min, float max);
float rand_normal(float mean, float std);

float distance(Point *p1, Point *p2);

void initialize_particle_filter(const Area *map, Particle *particles[], int particle_count);

void free_particles(Particle *particles[], int particle_count);

void predict_motion(const Point *v, const Area *map,
                    Particle *particles[], int particle_count,
                    float motion_noise_std, float dt
                );

void update_weights(Particle *particles[], int particle_count,
                    Point beacons[], int beacon_count,
                    float path_loss_exponent, float d0,
                    float RSSI_0, float RSSI_noise_std,
                    float measurement_RSSI[],
                    float temp_weights[]
                );

int binary_search(float *cumulative_weights, int size, float value);

void resample_particles(Particle *particles[], int particle_count, Particle *new_particles[]);

Point estimate_position(Particle *particles[], int particle_count);

// Simulate RSSI measurement (with noise)
void get_rssi_measurement(Point *true_pos, Point beacons[], float observed_rssi[]);

