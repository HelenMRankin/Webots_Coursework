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
#define PS_FRONT_LEFT "ps7"

// Define ground sensor
#define GS "gs1"

#define NUMBER_OF_INPUTS 3
#define NUM_SENSORS 5
#define INPUT_LAYER_NUMBER_OF_NEURONS 5
#define HIDDEN_LAYER_NUMBER_OF_NEURONS 5
#define OUTPUT_LAYER_NUMBER_OF_NEURONS 2

#define NEURON_INPUTS 3
#define NUMBER_OF_NEURONS 7
#define MAX_SPEED 1000

#define GENOTYPE_SIZE (NEURON_INPUTS*NUMBER_OF_NEURONS)
double genes[21];
double inputs[5];
double recurrent_inputs[HIDDEN_LAYER_NUMBER_OF_NEURONS];
WbDeviceTag sensors[NUM_SENSORS];  // proximity sensors
WbDeviceTag receiver;              // for receiving genes from Supervisor


// check if a new set of genes was sent by the Supervisor
// in this case start using these new genes immediately
void check_for_new_genes() {
  if(wb_receiver_get_data_size(receiver) == GENOTYPE_SIZE * sizeof(double)) {
    printf("Got genes\n");
    const double* weights = wb_receiver_get_data(receiver);
    int m;
  
    // Copy genes as an array in nn_weights format.
    for (m = 0; m < GENOTYPE_SIZE; ++m) {
        genes[m] = weights[m]; 
    }
 
  }
  
   // printf("Size received: %d \n", GENOTYPE_SIZE * sizeof(double));
    
    // copy new genes directly in the sensor/actuator matrix
    // we don't use any specific mapping nor left/right symmetry
    // it's the GA's responsability to find a functional mapping
//   memcpy(weights,nn_weights, 18);

    // prepare for receiving next genes packet
    wb_receiver_next_packet(receiver); 
}

// sguash the ANN output between -1 and 1.
double hyperbolic_tangent(double value) {
 // printf("Tangent: %f\n", value);
  return (1.0f - exp(- 2.0f * value)) / (1.0f + exp(-2.0f * value));
}

// Calculate the output based on weights evolved by GA.
double* evolve_neural_net() {
 
  int i,j;
  double input_layer_outputs[INPUT_LAYER_NUMBER_OF_NEURONS],
  hidden_layer_outputs[HIDDEN_LAYER_NUMBER_OF_NEURONS], 
  *output_layer_outputs;
  
  output_layer_outputs = malloc(OUTPUT_LAYER_NUMBER_OF_NEURONS);
  
  printf("#### PROCESS INPUT LAYER ####\n");
  // Input layer.
  for (i = 0; i < INPUT_LAYER_NUMBER_OF_NEURONS; ++i) {
    printf("out neuron: %i\n",i);
    double output = 0.0f;
    for (j = 0; j < NUMBER_OF_INPUTS; ++j) {
      int gene_number = NUMBER_OF_INPUTS * i + j;
      printf("   Gene number: %i\n", gene_number);
      if (i+j == 0) {
        output += genes[gene_number];
      }
      if(i+j == 6) {
        output += genes[gene_number];
      }
      else {
        printf("   Input neuron: %i\n", j-1+ i);
        output += genes[gene_number] * tanh(inputs[j-1+ i]);
      }        
    }
    
    input_layer_outputs[i] = tanh(output);
  }
  // Hidden layer.
  int  gene_offset = (NUMBER_OF_INPUTS ) * INPUT_LAYER_NUMBER_OF_NEURONS;
  printf("Gene offset: %i\n", gene_offset);
  int hidden_layer_inputs = INPUT_LAYER_NUMBER_OF_NEURONS + HIDDEN_LAYER_NUMBER_OF_NEURONS;
  printf("#### PROCESS HIDDEN LAYER ####\n");
  for (i = 0; i < HIDDEN_LAYER_NUMBER_OF_NEURONS; ++i) {
    printf("Hidden neuron %i\n", i);
    double output = 0.0f;
 
    for (j = 0 ; j < INPUT_LAYER_NUMBER_OF_NEURONS + 1; ++j) {  
      int gene_number = gene_offset + HIDDEN_LAYER_NUMBER_OF_NEURONS * i + i + j;
      printf("\tGene number %i\n", gene_number);  
      if (i+j == 0) {
        output += genes[gene_number];
        //printf("gene: [%d]\n", gene_offset + (hidden_layer_inputs) * i + i + j);
      }
      else {
        printf("\tInput neuron %i\n", j-1+ i);
        //printf("gene: [%d]\n", gene_offset + hidden_layer_inputs * i + i + j);
        output += genes[gene_number] * input_layer_outputs[j-1+ i];
      }        
    }
    
    // Elmar neural network implementation.
    int k;
    for (k = j ; k < HIDDEN_LAYER_NUMBER_OF_NEURONS + j ; ++k) {
      //printf("gene: [%d]\n", gene_offset + hidden_layer_inputs * i + i + k);
      output += genes[gene_offset + hidden_layer_inputs * i + i + k] * recurrent_inputs[k];      
    } 
    //printf("Break\n");
    hidden_layer_outputs[i] = tanh(output);
  }
  
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
        //printf("gene: [%d]\n", gene_offset + HIDDEN_LAYER_NUMBER_OF_NEURONS * i + i + j);
      }
      else {
        output += genes[gene_offset + HIDDEN_LAYER_NUMBER_OF_NEURONS * i + i + j] * hidden_layer_outputs[j-1];
        //printf("gene: [%d]\n", gene_offset + HIDDEN_LAYER_NUMBER_OF_NEURONS * i + i + j);
      }        
    }
    
    output_layer_outputs[i] = tanh(output);
  }
  return output_layer_outputs;
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
  
 // printf("Wheel speeds: %f, %f\n", wheel_speeds[0]*MAX_SPEED, wheel_speeds[1]*MAX_SPEED);
  // actuate e-puck wheels
  wb_differential_wheels_set_speed(wheel_speeds[0]*MAX_SPEED, wheel_speeds[1]*MAX_SPEED);
}

int main(int argc, const char *argv[]) {

printf("Start\n");
  wb_robot_init();  // initialize Webots
  memset(genes, 0.0, 21);
  // find simulation step in milliseconds (WorldInfo.basicTimeStep)
  int time_step = wb_robot_get_basic_time_step();
  char * sensor_names[NUM_SENSORS] = {PS_LEFT, PS_FRONT_LEFT, GS, PS_FRONT_RIGHT, PS_RIGHT};
  
printf("test 1\n");
  // find and enable all sensors
  int i;
  for (i = 0; i < NUM_SENSORS; i++) {
    sensors[i] = wb_robot_get_device(sensor_names[i]);
    wb_distance_sensor_enable(sensors[i], time_step);
  }
  printf("test 2\n");

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
