#include "particle_filter_static.h"

#include <time.h>
#include <stdio.h>
#include <stdlib.h>

#define M_PI 3.1415926

#include <math.h>

float rand_uniform(float min, float max) {
    return min + (max - min) * (rand() / (float)RAND_MAX);
}

float rand_normal(float mean, float std) {
    float u1, u2;
    do {
        u1 = rand_uniform(0.0f, 1.0f);
        u2 = rand_uniform(0.0f, 1.0f);
    } while (u1 <= 1e-12f);  // Avoid log(0)

    // Box-Muller transform
    float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
    return mean + std * z;
}

float distance(Point *p1, Point *p2) {
    float dx = p1->x - p2->x;
    float dy = p1->y - p2->y;
    return sqrtf(dx * dx + dy * dy);
}

void initialize_particle_filter(const Area *map, Particle *particles[], int particle_count) {
    for (int i = 0; i < particle_count; i++) {
        particles[i] = (Particle *)malloc(sizeof(Particle));
        if (!particles[i]) {
            perror("Failed to allocate particle");
            exit(EXIT_FAILURE);
        }
        
        particles[i]->pos.x = rand_uniform(map->min.x, map->max.x);
        particles[i]->pos.y = rand_uniform(map->min.y, map->max.y);
        particles[i]->weight = 1.0f / particle_count;
    }
}

void free_particles(Particle *particles[], int particle_count) {
    for (int i = 0; i < particle_count; i++) {
        free(particles[i]);
    }
}

// Move particles based on known velocity
void predict_motion(const Point *v, const Area *map, Particle *particles[], int particle_count, float motion_noise_std, float dt) {
    for (int i = 0; i < particle_count; i++) {
        // Add independent noise to each particle's motion
        float dx = v->x * dt + rand_normal(0.0f, motion_noise_std);
        float dy = v->y * dt + rand_normal(0.0f, motion_noise_std);

        particles[i]->pos.x += dx;
        particles[i]->pos.y += dy;

        // Clamp to map boundaries
        particles[i]->pos.x = fmaxf(map->min.x, fminf(map->max.x, particles[i]->pos.x));
        particles[i]->pos.y = fmaxf(map->min.y, fminf(map->max.y, particles[i]->pos.y));
    }
}

void update_weights(Particle *particles[], int particle_count,
                    Point beacons[], int beacon_count,
                    float path_loss_exponent, float d0,
                    float RSSI_0, float RSSI_noise_std,
                    float measurement_RSSI[],
                    float temp_weights[]
                ) {
    
    float max_log_likelihood = -INFINITY;
    float total_weight = 0.0f;

    // First pass: compute log-likelihoods and find maximum
    for (int i = 0; i < particle_count; i++) {
        float log_likelihood = 0.0f;

        for (int j = 0; j < beacon_count; j++) {
            float d = distance(&particles[i]->pos, &beacons[j]);
            if (d < 0.1f) d = 0.1f;  // Avoid division by very small numbers
            
            float expected_rssi = RSSI_0 - 10.0f * path_loss_exponent * log10f(d / d0);
            float rssi_diff = measurement_RSSI[j] - expected_rssi;
            log_likelihood += -0.5f * (rssi_diff * rssi_diff) / (RSSI_noise_std * RSSI_noise_std);
        }

        temp_weights[i] = log_likelihood;
        if (log_likelihood > max_log_likelihood) {
            max_log_likelihood = log_likelihood;
        }
    }

    // Second pass: compute normalized weights
    for (int i = 0; i < particle_count; i++) {
        particles[i]->weight = expf(temp_weights[i] - max_log_likelihood);
        total_weight += particles[i]->weight;
    }

    // Normalize weights
    if (total_weight > 1e-12f) {
        for (int i = 0; i < particle_count; i++) {
            particles[i]->weight /= total_weight;
        }
    } else {
        // Reset to uniform distribution if weights became too small
        float uniform_weight = 1.0f / particle_count;
        for (int i = 0; i < particle_count; i++) {
            particles[i]->weight = uniform_weight;
        }
    }
}

int binary_search(float *cumulative_weights, int size, float threshold) {
    int low = 0;
    int high = size - 1;

    while (low < high) {
        int mid = (low + high) / 2;
        if (threshold > cumulative_weights[mid]) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    return low;
}

void resample_particles(Particle *particles[], int particle_count, Particle *new_particles[]) {
    float cumulative_weights[particle_count];
    cumulative_weights[0] = particles[0]->weight;
    
    for (int i = 1; i < particle_count; i++) {
        cumulative_weights[i] = cumulative_weights[i - 1] + particles[i]->weight;
    }

    // Systematic resampling
    float step = 1.0f / particle_count;
    float u = rand_uniform(0.0f, step);
    
    for (int i = 0; i < particle_count; i++) {
        float threshold = u + i * step;


        int idx = binary_search(cumulative_weights, particle_count, threshold);

        new_particles[i]->pos = particles[idx]->pos;
        new_particles[i]->weight = 1.0f / particle_count;
    }

    // Replace old particles with new ones
    for (int i = 0; i < particle_count; i++) {
        particles[i] = new_particles[i];
    }
}

Point estimate_position(Particle *particles[], int particle_count) {
    Point estimate = {0.0f, 0.0f};
    
    for (int i = 0; i < particle_count; i++) {
        estimate.x += particles[i]->pos.x * particles[i]->weight;
        estimate.y += particles[i]->pos.y * particles[i]->weight;
    }
    
    return estimate;
}

#define NUM_PARTICLES 2750  // Increased for better estimation
#define NUM_BEACONS 4
#define RSSI_NOISE_STD 2.0f
#define RSSI0 -40.0f       // Reference RSSI at 1m
#define PATH_LOSS_EXPONENT 2.0f
#define REFERENCE_DISTANCE 1.0f
#define MOTION_NOISE_STD 0.01f
#define TIME_STEP 1.0f
#define SIMULATION_STEPS 50

// Simulate RSSI measurement (with noise)
void get_rssi_measurement(Point *true_pos, Point beacons[], float observed_rssi[]) {
    for (int j = 0; j < NUM_BEACONS; j++) {
        float d = distance(true_pos, &beacons[j]);
        if (d < 0.1f) d = 0.1f;  // Avoid log of very small numbers
        float rssi = RSSI0 - 10.0f * PATH_LOSS_EXPONENT * log10f(d / REFERENCE_DISTANCE);
        observed_rssi[j] = rssi + rand_normal(0.0f, RSSI_NOISE_STD);
    }
}

int main() {
    srand(time(NULL));

    // Beacon positions
    Point beacons[NUM_BEACONS] = {
        {0.0f, 0.0f}, {10.0f, 0.0f}, {0.0f, 10.0f}, {10.0f, 10.0f}
    };

    // Motion parameters
    const Point v = {0.2f, 0.1f};
    const Area map = {{0.0f, 0.0f}, {10.0f, 10.0f}};

    // Initialize particles
    Particle *particles[NUM_PARTICLES];
    initialize_particle_filter(&map, particles, NUM_PARTICLES);
    float *temp_weights = (float*)malloc(NUM_PARTICLES * sizeof(float));
    
    if (!temp_weights) {
        perror("Failed to allocate temporary weights");
        exit(EXIT_FAILURE);
    }

    Particle *new_particles[NUM_PARTICLES];

    for (int i = 0; i < NUM_PARTICLES; i++) {
        new_particles[i] = (Particle*)malloc(sizeof(Particle));
        if (!new_particles[i]) {
            perror("Failed to allocate new particle");
            exit(EXIT_FAILURE);
        }
    }
    

    // Simulation variables
    Point true_pos = {0.0f, 0.0f};
    Point estimated_trajectory[SIMULATION_STEPS];

    float d;
    float max = 0.0f;
    float min = 100.0f;
    float mean = 0.0f;

    clock_t start = clock();

    for (int t = 0; t < SIMULATION_STEPS; t++) {
        // Update true position
        true_pos.x += v.x * TIME_STEP;
        true_pos.y += v.y * TIME_STEP;
        
        // Ensure true position stays within bounds
        if (true_pos.x < map.min.x) true_pos.x = map.min.x;
        else if (true_pos.x > map.max.x) true_pos.x = map.max.x;

        if (true_pos.y < map.min.y) true_pos.y = map.min.y;
        else if (true_pos.y > map.max.y) true_pos.y = map.max.y;


        // Simulate RSSI measurements
        float observed_rssi[NUM_BEACONS];
        get_rssi_measurement(&true_pos, beacons, observed_rssi);

        // Particle filter steps
        predict_motion(&v, &map, particles, NUM_PARTICLES, MOTION_NOISE_STD, TIME_STEP);
        update_weights(particles, NUM_PARTICLES, beacons, NUM_BEACONS,
                     PATH_LOSS_EXPONENT, REFERENCE_DISTANCE,
                     RSSI0, RSSI_NOISE_STD, observed_rssi, temp_weights);
        resample_particles(particles, NUM_PARTICLES, new_particles);
        
        estimated_trajectory[t] = estimate_position(particles, NUM_PARTICLES);

        //Print results
        // printf("Step %2d: True (%.2f, %.2f), Estimated (%.2f, %.2f)\n",
        //        t, true_pos.x, true_pos.y,
        //        estimated_trajectory[t].x, estimated_trajectory[t].y);
        
        d = distance(&true_pos, &estimated_trajectory[t]);
        
        if (d > max) {
            max = d;
        }
        if (d < min) {
            min = d;
        }

        mean += d;
    }

    clock_t end = clock();
    double elapsed_ms = 1000.0 * (end - start) / CLOCKS_PER_SEC;
    printf("Elapsed time: %.2f ms\n", elapsed_ms);

    mean /= SIMULATION_STEPS;

    printf("Max error: %.2f, Min error: %.2f, Mean error: %.2f\n", max, min, mean);
    
    // Free allocated memory
    free_particles(particles, NUM_PARTICLES);
    free(temp_weights);

    return 0;
}