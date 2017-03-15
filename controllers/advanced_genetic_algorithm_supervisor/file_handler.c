#include <string.h>
#include <stdio.h>
#include "file_handler.h"
#include "genotype.h"
#include "population.h"

static const char *BEST_GENOTYPE_FILE = "best_genotype.txt";
static const char *BEST_POPULATION_FILE = "best_population.txt";
static const char *NEXT_POPULATION_FILE = "next_population.txt";

void write_next_population(Population p) {
  FILE *infile = fopen(NEXT_POPULATION_FILE, "w");
  if (! infile && p) {
    printf("unable to read %s\n", NEXT_POPULATION_FILE);
    return;
  }
  printf("Writing next population\n");
  
  write_population(p, infile);
  
  fflush(infile);
  fclose(infile);
}

void read_next_population(Population p) {
  FILE *infile = fopen(NEXT_POPULATION_FILE, "r");
  if (! infile) {
    printf("unable to read %s\n", NEXT_POPULATION_FILE);
    return;
  }
  
  read_population(p, infile);
  
  fflush(infile);
  fclose(infile);
}

void read_best_population(Population p) {
  FILE *infile = fopen(BEST_POPULATION_FILE, "r");
  if (! infile) {
    printf("unable to read %s\n", BEST_POPULATION_FILE);
    return;
  }
  
  read_population(p, infile);
  
  fflush(infile);
  fclose(infile);
}

void write_best_population(Population p) {
  FILE *infile = fopen(BEST_POPULATION_FILE, "w");
  if (! infile && p) {
    printf("unable to read %s\n", BEST_POPULATION_FILE);
    return;
  }
  printf("Writing new best population\n");
  
  write_population(p, infile);
  
  fflush(infile);
  fclose(infile);
}

void read_best_genotype(Genotype g) {
  FILE *infile = fopen(BEST_GENOTYPE_FILE, "r");
  if (! infile) {
    printf("unable to read %s\n", BEST_GENOTYPE_FILE);
    return;
  }

  genotype_fread(g, infile);
  fclose(infile);
}

void write_best_genotype(Genotype g) {
  FILE *outfile = fopen(BEST_GENOTYPE_FILE, "w");
  if (outfile&&g) {
    genotype_fwrite(g, outfile);
    fflush(outfile);
    fclose(outfile);
    printf("wrote best genotype with fitness %g into %s\n",  genotype_get_fitness(g), BEST_GENOTYPE_FILE);
  }
  else
    printf("unable to write %s\n", BEST_GENOTYPE_FILE);
}