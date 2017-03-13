//   Description:   Supervisor code for genetic algorithm

#include "genotype.h"
#include "population.h"
#include "file_handler.h"
#include <webots/supervisor.h>
#include <webots/robot.h>
#include <webots/emitter.h>
#include <webots/pen.h>
#include <webots/display.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
static const int POPULATION_SIZE = 50;
static const int NUM_GENERATIONS = 2;
static bool walls[14];
static bool walled_maze[12];
static bool unwalled_maze[12];

// must match the values in the advanced_genetic_algorithm.c code
static const int NUM_SENSORS = 7;
static const int NUM_WHEELS  = 3;
static const double EXPLORATION_BONUS = 20;
static const double TIME_PENALTY = 0.5;
static const double EXPLORATION_PENALTY = 20; 
#define GENOTYPE_SIZE (NUM_SENSORS * NUM_WHEELS)
#define FRONT_FACING 0
#define LEFT_FACING 1.5708
#define RIGHT_FACING 4.71239
#define BACK_FACING 3.1415
#define TWO_PI 6.283818
// index access
enum { X, Y, Z };

static int time_step;
static WbDeviceTag emitter;   // to send genes to robot
static WbDeviceTag display;   // to display the fitness evolution
static int display_width, display_height;

// the GA population
static Population population;

// for reading or setting the robot's position and orientation
static WbFieldRef robot_translation;
static WbFieldRef robot_rotation;
static double robot_trans0[3];  // a translation needs 3 doubles
static double robot_rot0[4];    // a rotation needs 4 doubles

// start with a demo until the user presses the 'O' key
// (change this if you want)
static bool demo = false;

double best_pop_fitness = 0;
double best_gen_fitness = 0;
Genotype fittest_overall;

void draw_scaled_line(int generation, double y1, double y2) {
  const double XSCALE = (double)display_width / NUM_GENERATIONS;
  const double YSCALE = 10.0;
  wb_display_draw_line(display, (generation - 0.5) * XSCALE, display_height - y1 * YSCALE,
    (generation + 0.5) * XSCALE, display_height - y2 * YSCALE);
}

double normalise(double radians) {
  double rotation_norm = fmod(radians,TWO_PI);
  if(rotation_norm<0) {
    rotation_norm+=TWO_PI;
  }
  return rotation_norm;
}

// plot best and average fitness
void plot_fitness(int generation, double best_fitness, double average_fitness) {
  static double prev_best_fitness = 0.0;
  static double prev_average_fitness = 0.0;
  if (generation > 0) {
    wb_display_set_color(display, 0xff0000); // red
    draw_scaled_line(generation, prev_best_fitness, best_fitness);

    wb_display_set_color(display, 0x00ff00); // green
    draw_scaled_line(generation, prev_average_fitness, average_fitness);
  }

  prev_best_fitness = best_fitness;
  prev_average_fitness = average_fitness;
}

bool in_range_of(double value, double target, double range) {
  double upper = normalise(target+range);
  double lower = normalise(target-range);
  if(lower + range >= TWO_PI) {
 //   printf("checking if %f is between %f or %f and %f\n", value, lower, -range, upper);
    return (lower <=value||-range<=value)&&upper>=value;
  }
//  printf("Checking if %f is between %f and %f\n", value, lower, upper);
  return(lower<=value && upper>=value);
}

void check_exploration(void * is_finished) {
  const double *location = wb_supervisor_field_get_sf_vec3f(robot_translation);
  const double * rotation = wb_supervisor_field_get_sf_rotation(robot_rotation);
  
 
  int index_x = 0;
  int index_z = 0;
  
  double x = location[2] * 10 ;
  double z = location[0] * 10;
  
  double z_round;
  
  index_x=floor(x);
   
 // printf("Index_x: %d\n", index_x);
  if(z<0) {
    index_z = floor(z);
    z_round=index_z;
    index_z+=7;
  } else {
    index_z = ceil(z);
    z_round=index_z;
    index_z+=4;
  }
  
  index_x += 6;
  index_x /=2;
//  printf("X: %f, index_x: %d\n", x, index_x);
  // Normalise rotation
  double rotation_norm = normalise(rotation[3]);
 // printf("Rotation: %f (normalised to %f)\n", rotation[3], rotation_norm);
  
  if(-1 <= z_round && z_round <= 1 && !walled_maze[index_x]) {
    if (in_range_of(rotation_norm,FRONT_FACING, 0.3) && index_x==0) { 
   //   printf("Seen front wall\n");
      walls[13]=true;
    }
    else if(in_range_of(rotation_norm, BACK_FACING, 0.3) && index_x==5) {
    //  printf("Seen back wall\n");
      walls[12]=true;
    }
    else if(in_range_of(rotation_norm, LEFT_FACING, 0.3)) {
  //    printf("Seen left wall\n");
      walls[2*index_x]=true;
    }
    else if(in_range_of(rotation_norm, RIGHT_FACING, 0.3)) {
    //  printf("Seen right wall\n");
      walls[2*index_x + 1] = true;
    }
  }
  else if(!walled_maze[index_z]){
    unwalled_maze[index_z] = true;
  }
  // Check if exploration is complete
  *((bool*) is_finished) = false;
  for(int i = 0; i< 14; i++ ) {
    *((bool*) is_finished) = *((bool*) is_finished) && walls[i];
  }
}

// run the robot simulation for the specified number of seconds
// may terminate after number of seconds if exploration complete
double run_seconds(double seconds) {
  printf("Start run\n");
  int i, n = 1000.0 * seconds / time_step;
  for (i = 0; i < n; i++) {
    if (demo ) {
      printf("Doing demo thing?\n");
      demo = false;
      return 60; // interrupt demo and start GA optimization
    }
    wb_robot_step(time_step);
   
    bool is_finished = false;
    check_exploration(&is_finished);
    if(is_finished) {
      printf("Run early\n");
      // return seconds taken
      return i*time_step/1000;
    }
    
  }
  printf("Run complete\n");
  return seconds;
}

void do_cleanup() {
  memset(walled_maze, false, sizeof(bool)* 12);
  memset(unwalled_maze, false, sizeof(bool)* 12);
  memset(walls, false, sizeof(bool) * 14);

}

// Fitness is calculated as 10 points per walled area explored, -1 point per second taken, 
double measure_fitness(double time) {
  double score = 0;
  
  int maze_explored_walled = 0;
  int maze_explored_unwalled = 0;
  
  for(int i=0; i<14; i++) {
    if(walls[i]==true) {
      maze_explored_walled++;
    }
    if(i<12 && unwalled_maze[i]==true) {
      maze_explored_unwalled++;
    }
  }
  
  score = maze_explored_walled * EXPLORATION_BONUS
        - time * TIME_PENALTY
        - maze_explored_unwalled * EXPLORATION_PENALTY;
        
  printf("Explored %i Walled and %i Unwalled areas in %g seconds. Score: %g\n", 
        maze_explored_walled, maze_explored_unwalled, time, score);
  do_cleanup();
  return score;
}

// evaluate one genotype at a time
void evaluate_genotype(Genotype genotype) {

  // send genotype to robot for evaluation
  wb_emitter_send(emitter, genotype_get_genes(genotype), GENOTYPE_SIZE * sizeof(double));

  // reset robot and load position
  wb_supervisor_field_set_sf_vec3f(robot_translation, robot_trans0);
  wb_supervisor_field_set_sf_rotation(robot_rotation, robot_rot0);
  
  double time_taken = run_seconds(15.0);
  
  // measure fitness
  double fitness = measure_fitness(time_taken);
  genotype_set_fitness(genotype, fitness);
  
  //printf("Clearing pen\n");
  //wb_supervisor_field_set_sf_float(pen_evaporation, 30);
 // run_seconds(10.0);
  

  printf("fitness: %g\n", fitness);
}

void run_optimization() {

  printf("---\n");
  printf("starting GA optimization ...\n");
  printf("population size is %d, genome size is %d\n", POPULATION_SIZE, GENOTYPE_SIZE);

  int i, j;
  for  (i = 0; i < NUM_GENERATIONS; i++) {
    for (j = 0; j < POPULATION_SIZE; j++) {
      printf("generation: %d, genotype: %d\n", i, j);

      // evaluate genotype
      Genotype genotype = population_get_genotype(population, j);
      evaluate_genotype(genotype);
    }

    double best_fitness = genotype_get_fitness(population_get_fittest(population));
    double average_fitness = population_compute_average_fitness(population);
    
    // display results
    plot_fitness(i, best_fitness, average_fitness);
    printf("best fitness: %g\n", best_fitness);
    printf("average fitness: %g\n", average_fitness);
    if(average_fitness>best_pop_fitness) {
      best_pop_fitness = average_fitness;
      write_best_population(population);
    }
    if(best_fitness>best_gen_fitness) {
      best_gen_fitness = best_fitness;
      fittest_overall = population_get_fittest(population);
    }
      
    // reproduce (but not after the last generation)
    if (i < NUM_GENERATIONS - 1)
      population_reproduce(population);
  }

  printf("GA optimization terminated.\n");

  // save fittest individual
  write_best_genotype(fittest_overall);

  population_destroy(population);
}


// show demo of the fittest individual
void run_demo() {

  printf("---\n");
  printf("running demo of best individual ...\n");
  printf("select the 3D window and push the 'O' key\n");
  printf("to start genetic algorithm optimization\n");

  Genotype genotype = genotype_create();
  read_best_genotype(genotype);

  while (demo)
    evaluate_genotype(genotype);
}

int main(int argc, const char *argv[]) {

  // initialize Webots
  wb_robot_init();
  
  memset(walled_maze, false, sizeof(bool));
  memset(unwalled_maze, false, sizeof(bool));
  memset(walls, false, sizeof(bool));
  // get simulation step in milliseconds
  time_step = wb_robot_get_basic_time_step();

  // the emitter to send genotype to robot
  emitter = wb_robot_get_device("emitter");
  
  // to display the fitness evolution
  display = wb_robot_get_device("display");
  display_width = wb_display_get_width(display);
  display_height = wb_display_get_height(display);
  wb_display_draw_text(display, "fitness", 2, 2);

  // initial population
  population = population_create(POPULATION_SIZE, GENOTYPE_SIZE);
 // read_best_population(population);
  // find robot node and store initial position and orientation
  WbNodeRef robot = wb_supervisor_node_get_from_def("ROBOT");
  robot_translation = wb_supervisor_node_get_field(robot, "translation");
  robot_rotation = wb_supervisor_node_get_field(robot, "rotation");

  memcpy(robot_trans0, wb_supervisor_field_get_sf_vec3f(robot_translation), sizeof(robot_trans0));
  memcpy(robot_rot0, wb_supervisor_field_get_sf_rotation(robot_rotation), sizeof(robot_rot0));

  if (demo)
    run_demo();

  // run GA optimization
  run_optimization();

  // cleanup Webots
  wb_robot_cleanup();
  return 0;  // ignored
}
