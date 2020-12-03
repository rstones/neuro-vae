import numpy as np
import random
random.seed()

class population:
    def __init__(self, pop_size, chromosomes, mutation_strength, crossover=1):
        self.pop_size = pop_size
        self.chromosomes = chromosomes
        self.mutation_strength = mutation_strength
        self.crossover = crossover
        self.males, self.females = self.initialize_population(self.pop_size, self.chromosomes)
        
        self.total_mean_fitness, self.total_max_fitness, self.mean_male_fitness, self.mean_female_fitness, self.max_male_fitness, self.max_female_fitness = self.pop_fitness(mode='train')
        self.total_mean_test_fitness, self.total_max_test_fitness, self.mean_male_test_fitness, self.mean_female_test_fitness, self.max_male_test_fitness, self.max_female_test_fitness = self.pop_fitness(mode='test')
        self.print_fitness()
        
    def pop_fitness(self, mode='train'):
        
        if mode=='train':
            arg = 'train fitness'
        elif mode=='test':
            arg = 'test fitness'
        
        male_fitness = [ i[arg] for i in self.males ]
        female_fitness = [ j[arg] for j in self.females ]
        
        mean_male = np.mean(male_fitness)
        mean_female = np.mean(female_fitness)
        max_male = np.max(male_fitness)
        max_female = np.max(female_fitness)
        
        total_mean_fitness = np.mean([female_fitness + male_fitness])
        total_max_fitness = np.max([female_fitness + male_fitness])

        return total_mean_fitness, total_max_fitness, mean_male, mean_female, max_male, max_female
    
    def make_individual(self, chromes, sex ):
        ind = {}
        
        ind['chromosomes'] = chromes
        ind['sex'] = sex
        ind['train fitness'] = self.get_fitness(ind, mode='train')
        ind['test fitness'] = self.get_fitness(ind, mode='test')
        return ind
        
    
    def initialize_population(self, pop_size, chromosomes):

        male_genes = np.stack( [ np.random.uniform( chromosomes['male'][i][0], chromosomes['male'][i][1], int(pop_size/2) ) for i in chromosomes['male'] ], axis=1)
        female_genes = np.stack( [ np.random.uniform( chromosomes['female'][i][0], chromosomes['female'][i][1], int(pop_size/2) ) for i in chromosomes['female'] ], axis=1)

        males=[]
        females=[]
        
        for row in male_genes:
            males.append(self.make_individual(list(row),'male')) 

        for row in female_genes:
            females.append(self.make_individual(list(row),'female'))
        return males, females
        
    def mate(self, ma, pa):
        if random.randint(0,1) == 1:
            ind = ma['chromosomes'][:self.crossover] + pa['chromosomes'][self.crossover:]
        else:
            ind = pa['chromosomes'][:self.crossover] + ma['chromosomes'][self.crossover:]
        
        sex = 'male' if random.randint(0,1) == 1 else 'female'
        
        #Mutation rate approaches 1 as population fitness stagnates
        if sex == 'male':
            mutation_rate =  ( 1 - (self.mean_male_fitness/self.max_male_fitness) ) * 0.1
            mutation_strength = 2 * self.mutation_strength
        else:
            mutation_rate = ( 1 - (self.mean_female_fitness/self.max_female_fitness) ) * 0.1
            mutation_strength = self.mutation_strength

        for gene in range(len(ind)):
            ind[gene] = self.mutate(ind[gene], mutation_rate, mutation_strength)
            
        sex = 'male' if random.randint(0,1) == 1 else 'female'

        individual = self.make_individual(ind, sex)
        
        return individual

    def mutate(self, gene, rate, strength):
        roll = random.random()
        if roll > 1-(rate/2):
            gene = gene + (roll * strength * gene)
        if roll < (rate/2):
            gene = gene - (1-roll * strength * gene)
        return gene
        
    
    def get_fitness(self, individual, mode='train'):
        
        var_list = quadratic_varlev_vector
        var_list[6:] = individual['chromosomes']
        
        lev_in, varlev_params = make_quadratic_varlev_params(var_list)
        
        #get max lev
        lev_out = []
        for bools in range(2,14):
            lev_out.append(individual['chromosomes'][0]*bools**2 + individual['chromosomes'][1]*bools + individual['chromosomes'][2])
        max_lev = np.max(lev_out)
        
        if mode=='train':
            mcaps = train_mcaps
            prices = train_prices
            topcaps = train_topcaps
        elif mode=='test':
            mcaps = test_mcaps
            prices = test_prices
            topcaps = test_topcaps
            
        test = strategy_wrapper(n_coins, pairs, weighter_sardine_hist, None, mcaps, prices, topcaps, dominance_pair=dominance_pair, short=False, plotting=[False, False], base='USD', cost=[0.0015,0.0013], leverage=lev_in, ourdom=n_coins, base_name_1='BTC', base_name_2='ETH', variable_lev=anchovies_quadratic_varlev, varlev_data=varlev_params)
        
        #If holdings ever goes negative discard
        if np.any([ i < 0 for i in test.holdings ]):
            f_score = 0
        else:
            f_score = test.holdings[-1] / max_lev
            
        return f_score
    
    def generation(self, male_prop, female_prop, mode='combined_evaluation'):
        male_fitness = [ i['train fitness'] for i in self.males ]
        female_fitness = [ j['train fitness'] for j in self.females ]
        total_fitness = female_fitness + male_fitness
        
        #generate fitness cutoffs
        if mode == 'combined_evaluation':
            male_cutoff = np.percentile(total_fitness, 100-male_prop)
            female_cutoff = np.percentile(total_fitness, 100-female_prop)
        else:
            male_cutoff = np.percentile(male_fitness, 100-male_prop)
            female_cutoff = np.percentile(female_fitness, 100-female_prop)

        #kill the boys and girls who fail
        for boy in self.males:
            #make sure we dont kill all males
            if len(self.males) == 1:
                break
                
            if boy['train fitness'] <= male_cutoff:
                self.males.pop(self.males.index(boy))
                
        for girl in self.females:
            #make sure we dont kill all females
            if len(self.females) == 1:
                break
                
            if girl['train fitness'] <= female_cutoff:
                self.females.pop(self.females.index(girl))
                
        #mate randomly among winners while pop_size below original 
        new_males = []
        new_females = []
        while len(new_males) + len(new_females) < self.pop_size:
            ma = random.choice(self.females)
            pa = random.choice(self.males)
            
            baby = self.mate(ma,pa)
            new_males.append(baby) if baby['sex'] == 'male' else new_females.append(baby)
            
        self.males = new_males
        self.females = new_females
        
        self.total_mean_fitness, self.total_max_fitness, self.mean_male_fitness, self.mean_female_fitness, self.max_male_fitness, self.max_female_fitness = self.pop_fitness(mode='train')
        self.total_mean_test_fitness, self.total_max_test_fitness, self.mean_male_test_fitness, self.mean_female_test_fitness, self.max_male_test_fitness, self.max_female_test_fitness = self.pop_fitness(mode='test')

        self.print_fitness()
        
    def print_fitness(self):
        print( 'total mean fitness: ' ,self.total_mean_fitness , '\ntotal max fitness: ',self.total_max_fitness)
        print( 'female mean fitness: ' ,self.mean_female_fitness , '\nfemale max fitness: ' ,self.max_female_fitness)
        print( 'male mean fitness: ',self.mean_male_fitness, '\nmale max fitness: ',self.max_male_fitness)
        print('\n')
        print( 'total mean test fitness: ' ,self.total_mean_test_fitness , '\ntotal max test fitness: ',self.total_max_test_fitness)
        print( 'female mean test fitness: ' ,self.mean_female_test_fitness , '\nfemale max test fitness: ' ,self.max_female_test_fitness)
        print( 'male mean test fitness: ',self.mean_male_test_fitness, '\nmale max test fitness: ',self.max_male_test_fitness)
        print('############################################')