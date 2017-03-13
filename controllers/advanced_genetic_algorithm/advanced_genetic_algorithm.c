// Description:   Robot execution code for genetic algorithm

#include <webots/robot.h>
#include <webots/differential_wheels.h>
#include <webots/receiver.h>
#include <webots/distance_sensor.h>
#include <assert.h>
#include <string.h> 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Define distance sensors
#define PS_FRONT_RIGHT "ps0"
#define PS_RIGHT "ps2"
#define PS_LEFT "ps5"
#define PS_BACK_RIGHT "ps3"
#define PS_BACK_LEFT "ps4"
#define PS_FRONT_LEFT "ps7"

// Define ground sensor
#define GS "gs1"

#define NUM_SENSORS 7
// set neuron numbers and inputs here
#define NUMBER_OF_INPUTS 7
#define INPUT_LAYER_NUMBER_OF_NEURONS 7
#define HIDDEN_LAYER_NUMBER_OF_NEURONS 7
#define OUTPUT_LAYER_NUMBER_OF_NEURONS 2

#define MAX_SPEED 1000
int inputs[NUMBER_OF_INPUTS];
double recurrent_inputs[HIDDEN_LAYER_NUMBER_OF_NEURONS];

#define GENOTYPE_SIZE ((NUMBER_OF_INPUTS + 1) * INPUT_LAYER_NUMBER_OF_NEURONS + (INPUT_LAYER_NUMBER_OF_NEURONS + 1) * HIDDEN_LAYER_NUMBER_OF_NEURONS + (HIDDEN_LAYER_NUMBER_OF_NEURONS * HIDDEN_LAYER_NUMBER_OF_NEURONS) + (HIDDEN_LAYER_NUMBER_OF_NEURONS + 1) * OUTPUT_LAYER_NUMBER_OF_NEURONS)
double genes[GENOTYPE_SIZE];
WbDeviceTag sensors[NUM_SENSORS];  // proximity sensors
WbDeviceTag receiver;              // for receiving genes from Supervisor

#define DEBUG_NN
#define DEBUG_GENES
int random_get_integer(int max) {
  return rand() % max;
}

double random_get_uniform() {

  return (double)(random_get_integer(60000)-30000)/(double)10000;
}

// check if a new set of genes was sent by the Supervisor
// in this case start using these new genes immediately
void check_for_new_genes() {
  if(wb_receiver_get_data_size(receiver) == GENOTYPE_SIZE * sizeof(double)) {
    const double* weights = wb_receiver_get_data(receiver);
    int m;
  
    // Copy genes as an array in nn_weights format.
    for (m = 0; m < GENOTYPE_SIZE; ++m) {
        genes[m] = weights[m]; 
    }
  }
  
  // prepare for receiving next genes packet
  wb_receiver_next_packet(receiver); 
}

// sguash the ANN output between -1 and 1.
double hyperbolic_tangent(double value) {
  printf("Tangent: %f\n", value);
  return (1.0f - exp(- 2.0f * value)) / (1.0f + exp(-2.0f * value));
}

// Calculate the output based on weights evolved by GA.
double* evolve_neural_net() {
 
  int i,j;
  static double input_layer_outputs[INPUT_LAYER_NUMBER_OF_NEURONS],
  hidden_layer_outputs[HIDDEN_LAYER_NUMBER_OF_NEURONS], 
  output_layer_outputs[OUTPUT_LAYER_NUMBER_OF_NEURONS];
  
  #ifdef DEBUG_GENES
  for (i = 0; i < GENOTYPE_SIZE; ++i) {
    printf("Genes input[%d]: %f\n", i, genes[i]); 
  }
  #endif
  
  #ifdef DEBUG_NN
  printf("PROCESS INPUT LAYER \n");
  #endif
  
  // Input layer.
  for (i = 0; i < INPUT_LAYER_NUMBER_OF_NEURONS; ++i) {
    
    double output = 0.0f;
    for (j = 0; j < NUMBER_OF_INPUTS + 1; ++j) {
      int gene_index = NUMBER_OF_INPUTS * i + i + j;
      double input_value = 1;
      
      if (j > 0) {
        input_value = inputs[j-1];
      }
      
      output += genes[NUMBER_OF_INPUTS * i + i + j] * input_value;
      printf("Gene: [%d], Neuron in: [%d], Neuron out: [%d], weight: %f, value: %f \n", gene_index, j-1, i, genes[gene_index], input_value );
    }
    
    input_layer_outputs[i] = hyperbolic_tangent(output);
  }
  //printf("input_layer_outputs[0]: %f\n", input_layer_outputs[0]);
  //printf("input_layer_outputs[1]: %f\n", input_layer_outputs[1]);
  
  // Hidden layer.
  int  gene_offset = (NUMBER_OF_INPUTS + 1) * INPUT_LAYER_NUMBER_OF_NEURONS;
  int hidden_layer_inputs = INPUT_LAYER_NUMBER_OF_NEURONS + HIDDEN_LAYER_NUMBER_OF_NEURONS;
  for (i = 0; i < HIDDEN_LAYER_NUMBER_OF_NEURONS; ++i) {
    
    double output = 0.0f;
    for (j = 0 ; j < INPUT_LAYER_NUMBER_OF_NEURONS + 1; ++j) {
      if (j == 0) {
        output += genes[gene_offset + hidden_layer_inputs * i + i + j];
   //     printf("gene: [%d]\n", gene_offset + (hidden_layer_inputs) * i + i + j);
      }
      else {
     //   printf("gene: [%d]\n", gene_offset + hidden_layer_inputs * i + i + j);
        output += genes[gene_offset + hidden_layer_inputs * i + i + j] * input_layer_outputs[j-1];
      }        
    }
    
    // Elmar neural network implementation.
    int k;
    for (k = j ; k < HIDDEN_LAYER_NUMBER_OF_NEURONS + j ; ++k) {
      
      printf("gene: [%d] (genotype_size: %i)\n", gene_offset + hidden_layer_inputs * i + i + k, GENOTYPE_SIZE);
      output += genes[gene_offset + hidden_layer_inputs * i + i + k] * recurrent_inputs[k];      
    } 
    hidden_layer_outputs[i] = hyperbolic_tangent(output);
  }
  //printf("hidden_layer_outputs[0]: %f\n", hidden_layer_outputs[0]);
  //printf("hidden_layer_outputs[1]: %f\n", hidden_layer_outputs[1]);
  
  // Save hidden layer outputs as recurrent inputs to be used next time.
  for (i = 0 ; i < HIDDEN_LAYER_NUMBER_OF_NEURONS ; ++i) {
     recurrent_inputs[i] = hidden_layer_outputs[i];    
  } 
  
  // Output layer.
  gene_offset += (INPUT_LAYER_NUMBER_OF_NEURONS + 1) * HIDDEN_LAYER_NUMBER_OF_NEURONS 
  + (HIDDEN_LAYER_NUMBER_OF_NEURONS * HIDDEN_LAYER_NUMBER_OF_NEURONS);
      
  for (i = 0; i < OUTPUT_LAYER_NUMBER_OF_NEURONS; ++i) {
    
    double output = 0.0f;
    for (j = 0; j < HIDDEN_LAYER_NUMBER_OF_NEURONS + 1; ++j) {
      if (j == 0) {
        output += genes[gene_offset + HIDDEN_LAYER_NUMBER_OF_NEURONS * i + i + j];
    //    printf("gene: [%d]\n", gene_offset + HIDDEN_LAYER_NUMBER_OF_NEURONS * i + i + j);
      }
      else {
        output += genes[gene_offset + HIDDEN_LAYER_NUMBER_OF_NEURONS * i + i + j] * hidden_layer_outputs[j-1];
     //   printf("gene: [%d]\n", gene_offset + HIDDEN_LAYER_NUMBER_OF_NEURONS * i + i + j);
      }        
    }
    
    output_layer_outputs[i] = hyperbolic_tangent(output);
  }
  return output_layer_outputs;
}

// Initialize hidden layer recurrent neural net inputs.
void init_recurrent_inputs() {
  int i;
  for (i = 0; i < HIDDEN_LAYER_NUMBER_OF_NEURONS; ++i) {
    recurrent_inputs[i] = random_get_uniform();
    //printf("Recurrent input[%d]: %f\n", i, recurrent_inputs[i]); 
  }
}

// Get input, evolve NN and move based on output
void sense_compute_and_actuate() {
  // read sensor values
  int i;
 // printf("Reading sensors\n");
  for (i = 0; i < NUM_SENSORS; i++) {
    inputs[i] = wb_distance_sensor_get_value(sensors[i]);
  }
  
  double * wheel_speeds = evolve_neural_net();
  
  printf("Wheel speeds: %f, %f\n", wheel_speeds[0]*MAX_SPEED, wheel_speeds[1]*MAX_SPEED);
  // actuate e-puck wheels
  wb_differential_wheels_set_speed(wheel_speeds[0]*MAX_SPEED, wheel_speeds[1]*MAX_SPEED);
}

int main(int argc, const char *argv[]) {
printf("Genotype size in robot: %i\n", GENOTYPE_SIZE);
  wb_robot_init();  // initialize Webots
  memset(genes, 0.0, 21);
  memset(recurrent_inputs, 0, HIDDEN_LAYER_NUMBER_OF_NEURONS);
  // find simulation step in milliseconds (WorldInfo.basicTimeStep)
  int time_step = wb_robot_get_basic_time_step();
  char * sensor_names[NUM_SENSORS] = {PS_LEFT, PS_FRONT_LEFT, PS_BACK_LEFT, GS, PS_BACK_RIGHT, PS_FRONT_RIGHT, PS_RIGHT};
  

  // find and enable all sensors
  int i;
  for (i = 0; i < NUM_SENSORS; i++) {
    sensors[i] = wb_robot_get_device(sensor_names[i]);
    wb_distance_sensor_enable(sensors[i], time_step);
  }
  init_recurrent_inputs();
  // find and enable receiver
  receiver = wb_robot_get_device("receiver");
  wb_receiver_enable(receiver, time_step);
  printf("test3\n");
  // run until simulation is restarted
  while (wb_robot_step(time_step) != -1) {
    check_for_new_genes();
    sense_compute_and_actuate();
  }

  wb_robot_cleanup();  // cleanup Webots
  return 0;            // ignored
}
