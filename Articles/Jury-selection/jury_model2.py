# Jury simulation model, using SimPy
# www.solvermax.com

import simpy
import random
import pandas as pd
import numpy as np
import logging

class gbl:
    # Scenario parameters
    pool_jurors_per_trial = 3  # Number of pool people summoned per trial
    pool_min = 5  # Minimium number of pool people summoned
    assigned_jurors_per_trial = 8  # Number of assigned people summoned per trial (after challenge process)

    # Time constants
    elapsed_day = 60 * 8  # Minutes per day
    elapsed_week = elapsed_day * 5  # Minutes per week

    # Process timing. Base time unit is minutes
    trial_week = 6 * elapsed_week  # Start of trial week
    reporting_time = 7 * elapsed_week  # Start of final reporting on trial outcomes
    duration = 8 * elapsed_week  # End time for the simulation
    show_cutoff = 60  # Must show before this time (from start of trial week)

    assign_generate_start = 0  # Time before starting process for assigned
    assigned_mean_response_time = 1  # Time to ask staff to be excused
    assigned_mean_summon_time = 1  # Time to do summoning
    assign_summon = 3 * elapsed_day  # Start of summon process for assigned
    assign_response = 3 * elapsed_week  # Start of excuse process for assigned
    assign_empanel_gap = 60  # Time between sign-in and empanelling
    
    pool_generate = 1 * elapsed_day
    pool_mean_response_time = 4  # Time to ask staff to be excused
    pool_mean_summon_time = 1  # Time to do summoning
    pool_summon = 2 * elapsed_day  # Start of summon process for pool
    pool_response = 3 * elapsed_week  # Start of excuse process for pool
    pool_empanel_gap = 120  # Time between sign-in and empanelling
        
    # Trial constants
    required_jurors_per_trial = 12  # Number of people on a jury
    num_courtrooms = 9  # Number of courtrooms available for trials
    mean_trials_per_week = 3  # Average number of trials in the week we're scheduling
    challenge_max_pool = 8  # Maximum number of candidates that can challenged per trial, if using pool only
    challenge_max_mixed = 2  # Maximum number of candidates that can challenged per trial, if using mix of assigned and pool only
    
    # Juror constants
    assigned_staff_excuse_rate = 25  # % summoned assign people who ask to be excused by staff
    assigned_no_show = 20  # % summoned assign people who are no show

    pool_staff_excuse_rate = 25  # % summoned pool people who ask to be excused by staff
    pool_no_show = 20  # % summoned pool people who are no show
    pool_judge_excuse_rate = 10  # % of people in empanelling process who ask Judge to be excused
    pool_challenge_rate = 25  # % of people in empanelling process who are challenged by lawers

    # Model constants
    num_iterations = 1  # Number of weeks to run
    iteration_update = 5000  # Update iteration count every n iterations
    use_seed = True  # Use random seed
    log_level = 'debug'  # Set logging level: debug, info, warning, error

class Assigned_Juror:  # People assigned to the jury for a specific trial
    def __init__(self, juror_id):
        self.id = juror_id
        self.trial = 0  # Trial assigned to. May not be empanelled
        self.summoned = False
        self.staff_excused = False
        self.show = False
        self.empanelled = False
        self.extra = False

class Pool_Juror:  # People in a pool for additional to juries
    def __init__(self, juror_id):
        self.id = juror_id
        self.trial = 0  # Trial empanelled on
        self.summoned = False
        self.staff_excused = False
        self.show = False
        self.challenged = False
        self.judge_excused = False
        self.empanelled = False
        self.extra = False
        
class Model:
    def __init__(self, iteration_number):
        self.env = simpy.Environment()
        self.trial_challenged = {}
        self.trial_empanelled = {}
        self.trial_status = {}
        self.assigned_counter = 0
        self.pool_counter = 0
        self.staff = simpy.Resource(self.env, capacity=1)
        self.iteration_number = iteration_number
        self.num_assigned = 0
        self.results = {}
        self.mean_q_time_nurse = 0
        self.jury_box = [simpy.Container(self.env, init=0, capacity=gbl.required_jurors_per_trial) for _ in range(gbl.num_courtrooms + 1)]

    def assign_generate(self, num_trials):  # Generate each assigned person
        yield self.env.timeout(gbl.assign_generate_start - self.env.now)  # Start of process
        self.num_assigned = num_trials * gbl.assigned_jurors_per_trial
        self.results['a_initial'] += self.num_assigned
        curr_trial = 0
        for j in range(self.num_assigned):
            if j % gbl.assigned_jurors_per_trial == 0:
                curr_trial += 1
            self.assigned_counter += 1
            assigned = Assigned_Juror(self.assigned_counter)
            assigned.trial = curr_trial
            logger.debug(f'Assigned {assigned.id:>4,} for trial {assigned.trial} created at {self.env.now:>8,.1f} ({convert_time(self.env.now, 'days')})')
            self.env.process(self.assign_process(assigned, num_trials))
            yield self.env.timeout(0)

    def assign_process(self, assigned, num_trials):  # Process each assigned person
        yield self.env.timeout(gbl.assign_summon - self.env.now)  # Advance clock to summon process
        self.env.process(self.assign_summon(assigned))
        yield self.env.timeout(gbl.assign_response - self.env.now)  # Advance clock to responses
        self.env.process(self.assign_staff_excuse(assigned))
        yield self.env.timeout(gbl.trial_week - self.env.now)  # Advance clock to trial week
        self.env.process(self.assign_show(assigned))
        yield self.env.timeout(gbl.trial_week + gbl.show_cutoff + gbl.assign_empanel_gap - self.env.now)  # Advance clock to empanelling time
        self.env.process(self.assign_empanel(assigned, num_trials))

    def assign_summon(self, assigned):
        summon_time = random.expovariate(1.0 / gbl.assigned_mean_summon_time)
        yield self.env.timeout(summon_time)
        with self.staff.request() as req:  # Summon each assigned person, if no already challenged
            yield req
            start = self.env.now
            process_time = np.random.lognormal(mean=1.0, sigma=0.5)
            end = start + process_time
            assigned.summoned = True
            logger.debug(f'Assigned {assigned.id:>4,} summoned at {start:>8,.1f}, took {process_time:>8,.1f}, end at {end:>8,.1f} ({convert_time(end, 'days')})')
            yield self.env.timeout(process_time)

    def assign_staff_excuse(self, assigned):
        if assigned.summoned:
            response_time = random.expovariate(1.0 / gbl.assigned_mean_response_time)  # Some summoned assigned people ask staff to be excused
            yield self.env.timeout(response_time)
            with self.staff.request() as req:
                yield req
                excused_time = self.env.now
                if np.random.uniform(0, 1) <= gbl.assigned_staff_excuse_rate / 100:  # 
                    assigned.staff_excused = True
                    self.results['a_excused'] += 1
                    logger.debug(f'Assigned {assigned.id:>4,} excused by staff at {excused_time:>8,.1f} ({convert_time(excused_time, 'days')})')
                yield self.env.timeout(0)

    def assign_show(self, assigned):
        show_time = min(gbl.show_cutoff, np.random.lognormal(mean=2.0, sigma=1.0))
        yield self.env.timeout(show_time)
        if assigned.summoned and not assigned.staff_excused:
            with self.staff.request() as req:
                yield req
                if np.random.uniform(0, 1) <= 1 - (gbl.assigned_no_show / 100):
                    assigned.show = True
                    self.results['a_show'] += 1
                    logger.debug(f'Assigned {assigned.id:>4,} showed up at {self.env.now:>8,.1f}')
                yield self.env.timeout(0)
            
    def assign_empanel(self, assigned, num_trials):
        empanel_time = np.random.lognormal(mean=1.0, sigma=1.0)
        yield self.env.timeout(empanel_time)
        with self.staff.request() as req:
            yield req
            curr_trial = assigned.trial
            if assigned.show:
                if self.trial_status[curr_trial] == False:
                    assigned.empanelled = True
                    self.trial_empanelled[curr_trial] += 1
                    yield self.jury_box[curr_trial].put(1)
                    self.results['a_empanelled'] += 1
                    logger.debug(f'Assigned {assigned.id:>4,} empanelled at {self.env.now:>8,.1f} on trial {curr_trial} which has {self.trial_empanelled[curr_trial]:>2} jurors')
                    if self.trial_empanelled[curr_trial] >= gbl.required_jurors_per_trial:
                        self.trial_status[curr_trial] = True
                        logger.debug(f'Trial {curr_trial} fully empanelled with assigned only at {self.env.now:>8,.1f} ({convert_time(self.env.now, 'days')})')
                        self.results['Success'] += 1
                else:
                    assigned.extra = True
                    logger.debug(f'Assigned {assigned.id:>4,} extra for trial {curr_trial} which has {self.trial_empanelled[curr_trial]} jurors')
            yield self.env.timeout(0)

    def pool_generate(self, num_trials):  # Generate each pool person
        yield self.env.timeout(gbl.pool_generate - self.env.now)  # Advance clock
        self.pool_counter = self.num_assigned  # Start pool id after end of assigened id
        if num_trials == 0:
            self.num_pool = 0
        else:
            self.num_pool = max(gbl.pool_min, num_trials * gbl.pool_jurors_per_trial)
        self.results['p_initial'] += self.num_pool
        for j in range(self.num_pool):
            self.pool_counter += 1
            pool = Pool_Juror(self.pool_counter)
            logger.debug(f'Pool     {pool.id:>4,} created at {self.env.now:>8,.1f} ({convert_time(self.env.now, 'days')})')
            self.env.process(self.pool_process(pool, num_trials))
            yield self.env.timeout(0)
        yield self.env.timeout(gbl.reporting_time - self.env.now)  # Advance clock to reporting time
        self.env.process(self.pool_report(num_trials))

    def pool_process(self, pool, num_trials):  # Process each pool person
        yield self.env.timeout(gbl.pool_summon - self.env.now)  # Advance clock to summon process
        self.env.process(self.pool_summon(pool))
        yield self.env.timeout(gbl.pool_response - self.env.now)  # Advance clock to excuse process
        self.env.process(self.pool_staff_excuse(pool))
        yield self.env.timeout(gbl.trial_week - self.env.now)  # Advance clock to trial week
        self.env.process(self.pool_show(pool))
        yield self.env.timeout(gbl.trial_week + gbl.show_cutoff + gbl.pool_empanel_gap - self.env.now)  # Advance clock to empanelling time
        self.env.process(self.pool_empanel(pool, num_trials))
        
    def pool_summon(self, pool):
        summon_time = random.expovariate(1.0 / gbl.pool_mean_summon_time)  # 
        yield self.env.timeout(summon_time)
        with self.staff.request() as req:  # Summon each pool person
            yield req
            start = self.env.now
            process_time = np.random.lognormal(mean=1.0, sigma=0.5)
            end = start + process_time
            pool.summoned = True
            logger.debug(f'Pool     {pool.id:>4,} summoned at {start:>8,.1f}, took {process_time:>8,.1f}, end at {end:>8,.1f} ({convert_time(end, 'days')})')
            yield self.env.timeout(process_time)

    def pool_staff_excuse(self, pool):
        if pool.summoned:
            response_time = random.expovariate(1.0 / gbl.pool_mean_response_time) # Some summoned pool people ask staff to be excused
            yield self.env.timeout(response_time)
            with self.staff.request() as req:
                yield req
                logger.debug(f'Pool     {pool.id:>4,} responded at {self.env.now:>8,.1f} ({convert_time(self.env.now, 'days')})')
                excused_time = self.env.now
                if np.random.uniform(0, 1) <= gbl.pool_staff_excuse_rate / 100:
                    pool.staff_excused = True
                    self.results['p_excused'] += 1
                    logger.debug(f'Pool     {pool.id:>4,} excused by staff at {excused_time:>8,.1f} ({convert_time(excused_time, 'days')})')
                yield self.env.timeout(0)

    def pool_show(self, pool):
        show_time = min(gbl.show_cutoff, np.random.lognormal(mean=2.0, sigma=1.0))
        yield self.env.timeout(show_time)
        if pool.summoned and not pool.staff_excused:
            with self.staff.request() as req:
                yield req
                if np.random.uniform(0, 1) <= 1 - (gbl.pool_no_show / 100):
                    pool.show = True
                    self.results['p_show'] += 1
                    logger.debug(f'Pool     {pool.id:>4,} showed up at {self.env.now:>8,.1f}')
                else:
                    logger.debug(f'Pool     {pool.id:>4,} no show at {self.env.now:>8,.1f}')
                yield self.env.timeout(0)
            
    def pool_empanel(self, pool, num_trials):
        empanel_time = np.random.lognormal(mean=1.0, sigma=1.0)
        yield self.env.timeout(empanel_time)
        with self.staff.request() as req:
            yield req
            if pool.show and not pool.empanelled:
                for curr_trial in range(1, num_trials + 1):
                    if not self.trial_status[curr_trial]:
                        yield self.env.process(self.pool_decision(pool, curr_trial))
                        if pool.empanelled:
                            break

    def pool_decision(self, pool, curr_trial):
        pool.judge_excused = False  # Reset excused and challenged because still eligible for other trials if excused or challenged for a trial
        pool.challenged = False
        challenge_max = gbl.challenge_max_mixed if gbl.assigned_jurors_per_trial > 0 else gbl.challenge_max_pool

        if np.random.uniform(0, 1) <= gbl.pool_judge_excuse_rate / 100:
            pool.judge_excused = True
            logger.debug(f'Pool     {pool.id:>4,} excused by Judge at {self.env.now:>8,.1f} ({convert_time(self.env.now, 'days')})')
    
        if not pool.judge_excused and (self.trial_challenged[curr_trial] < challenge_max):
            if np.random.uniform(0, 1) <= gbl.pool_challenge_rate / 100:
                pool.challenged = True
                self.trial_challenged[curr_trial] += 1
                logger.debug(f'Pool     {pool.id:>4,} challenged by lawyer at {self.env.now:>8,.1f} ({convert_time(self.env.now, 'days')})')

        if not pool.judge_excused and not pool.challenged:
            yield self.jury_box[curr_trial].put(1)
            self.trial_empanelled[curr_trial] += 1
            pool.empanelled = True
            self.results['p_empanelled'] += 1
            pool.trial = curr_trial
            logger.debug(f'Pool     {pool.id:>4,} empanelled at {self.env.now:>8,.1f} on trial {curr_trial} which has {self.trial_empanelled[curr_trial]:>2} jurors')
            if self.jury_box[curr_trial].level >= self.jury_box[curr_trial].capacity:
                self.trial_status[curr_trial] = True
                logger.debug(f'Trial {curr_trial} fully empanelled with mix of assigned and pool at {self.env.now:>8,.1f} ({convert_time(self.env.now, 'days')})')
                self.results['Success'] += 1

    def pool_report(self, num_trials):  # Report on trial success or failure
        with self.staff.request() as req:
            yield req
            for trial in range(1, num_trials+1):
                status = self.trial_status[trial]
                status_text = 'success' if status else 'fail'
                logger.debug(f'Trial {trial} status: {status_text}')
            
    def generate_trials(self):  # Generate the number of trials for the scheduled week, capped
        num_trials = 1 #min(np.random.poisson(gbl.mean_trials_per_week), gbl.num_courtrooms)
        logger.debug(f'Generated {num_trials} {'trial' if num_trials == 1 else 'trials'} for the scheduled week')
        self.trial_challenged = {t: 0 for t in range(1, num_trials+1)}
        self.trial_empanelled = {t: 0 for t in range(1, num_trials+1)}
        self.trial_status = {t: False for t in range(1, num_trials+1)}
        return num_trials
    
    def iteration(self):
        assigned_list = []
        num_trials = self.generate_trials()
        self.results['Trials'] = num_trials
        self.results['Success'] = 0
        self.results['a_initial'] = 0
        self.results['a_excused'] = 0
        self.results['a_show'] = 0
        self.results['a_empanelled'] = 0
        self.results['p_initial'] = 0
        self.results['p_excused'] = 0
        self.results['p_show'] = 0
        self.results['p_empanelled'] = 0
        self.env.process(self.assign_generate(num_trials))
        self.env.process(self.pool_generate(num_trials))
        self.env.run(until=gbl.duration)
        return self.results

class Simulation:
    def  __init__(self):
        self.iteration_results = pd.DataFrame()
        self.iteration_results['Iteration'] = [0]
        self.iteration_results.set_index('Iteration', inplace=True)  # Index by iteration. Add other columns dynamically. Need to initialize each column

    def run_simulation(self):
        for iteration_number in range(1, gbl.num_iterations+1):
            logger.debug(f'\nIteration {iteration_number}')  # Print iteration when debugging
            if iteration_number % gbl.iteration_update == 0:
                print(f'Iteration: {iteration_number:,.0f}')  # Print iteration when in batch mode
            sim_model = Model(iteration_number)
            results = sim_model.iteration()
            self.collate_results(iteration_number, results)

    def collate_results(self, iteration_number, results):
        self.iteration_results.at[iteration_number, 'Trials'] = results['Trials']
        self.iteration_results.at[iteration_number, 'Success'] = results['Success']
        self.iteration_results.at[iteration_number, 'Fail'] = results['Trials'] - results['Success']
        self.iteration_results.at[iteration_number, 'a_initial'] = results['a_initial']
        self.iteration_results.at[iteration_number, 'a_excused'] = results['a_excused']
        self.iteration_results.at[iteration_number, 'a_noshow'] = results['a_initial'] - results['a_excused'] - results['a_show']
        self.iteration_results.at[iteration_number, 'a_show'] = results['a_show']
        self.iteration_results.at[iteration_number, 'a_empanelled'] = results['a_empanelled']
        self.iteration_results.at[iteration_number, 'a_extra'] = results['a_show'] - results['a_empanelled']
        self.iteration_results.at[iteration_number, 'p_initial'] = results['p_initial']
        self.iteration_results.at[iteration_number, 'p_excused'] = results['p_excused']
        self.iteration_results.at[iteration_number, 'p_noshow'] = results['p_initial'] - results['p_excused'] - results['p_show']
        self.iteration_results.at[iteration_number, 'p_show'] = results['p_show']
        self.iteration_results.at[iteration_number, 'p_empanelled'] = results['p_empanelled']
        self.iteration_results.at[iteration_number, 'p_extra'] = results['p_show'] - results['p_empanelled']
        
    def print_sim_results(self):
        print('\nSimulation results')
        print('------------------\n')
        print(f'Weeks: {gbl.num_iterations:,.0f}')
        print(f'Pool per trial: {gbl.pool_jurors_per_trial}, minimum pool: {gbl.pool_min}, assigned per trial: {gbl.assigned_jurors_per_trial}\n')

        total_columns = self.iteration_results.sum()
        a_i = total_columns['a_initial']
        a_e = total_columns['a_excused']
        a_n = total_columns['a_noshow']
        a_s = total_columns['a_show']
        a_m = total_columns['a_empanelled']
        em_a_pct = a_m / a_s if a_s > 0 else 0
        a_x = total_columns['a_extra']
        ex_a_pct = (a_s - a_m) / a_s if a_s > 0 else 0
        p_i = total_columns['p_initial']
        p_e = total_columns['p_excused']
        p_n = total_columns['p_noshow']
        p_s = total_columns['p_show']
        p_m = total_columns['p_empanelled']
        em_p_pct = p_m / p_s if p_s > 0 else 0
        p_x = total_columns['p_extra']
        ex_p_pct = (p_s - p_m) / p_s if p_s > 0 else 0
        em_pct_tot = (a_m + p_m) / (a_s + p_s) if (a_s + p_s) > 0 else 0
        ex_pct_tot = ((a_s + p_s) - (a_m + p_m)) / ((a_s + p_s)) if (a_s + p_s) > 0 else 0
        t_i, t_e, t_n, t_s, t_m, t_ex = a_i + p_i, a_e + p_e, a_n + p_n, a_s + p_s, a_m + p_m, (a_s + p_s) - (a_m + p_m)        
        print(f'Jurors         Initial       Excused       No show          Show             Empanelled                 Extra')
        print(f'-------------------------------------------------------------------------------------------------------------')
        print(f'Assigned   {a_i:>11,.0f}   {a_e:>11,.0f}   {a_n:>11,.0f}   {a_s:>11,.0f}   {a_m:>11,.0f} ({em_a_pct:>6.1%})  {a_s - a_m:>11,.0f} ({ex_a_pct:>6.1%})')
        print(f'Pool       {p_i:>11,.0f}   {p_e:>11,.0f}   {p_n:>11,.0f}   {p_s:>11,.0f}   {p_m:>11,.0f} ({em_p_pct:>6.1%})  {p_s - p_m:>11,.0f} ({ex_p_pct:>6.1%})')
        print(f'-------------------------------------------------------------------------------------------------------------')
        print(f'Total      {t_i:>11,.0f}   {t_e:>11,.0f}   {t_n:>11,.0f}   {t_s:>11,.0f}   {t_m:>11,.0f} ({em_pct_tot:>6.1%})  {t_ex:>11,.0f} ({ex_pct_tot:>6.1%})')

        t_s_t, t_f_t, t_t_t = total_columns['Success'], total_columns['Fail'], total_columns['Trials']
        t_success_pct = (t_s_t / t_t_t) if t_t_t > 0 else 0
        t_fail_pct = (t_f_t / t_t_t) if t_t_t > 0 else 0
        total_pct = t_t_t/(t_s_t + t_f_t) if (t_s_t + t_f_t) > 0 else 0
        print(f'\nTrials                  Success                    Failure                  Total')
        print(f'---------------------------------------------------------------------------------')
        for t in range(1, gbl.num_courtrooms + 1):
            successes = self.iteration_results.groupby('Trials')['Success'].sum().get(t, 0)
            fails = self.iteration_results.groupby('Trials')['Fail'].sum().get(t, 0)
            total_trials = successes + fails
            success_pct = successes / total_trials if total_trials > 0 else 0
            failure_pct = fails / total_trials if total_trials > 0 else 0
            row_pct = total_trials/t_t_t if t_t_t > 0 else 0
            print(f'{t:>6,.0f} {successes:>11,.0f} ({success_pct:>10.5%})   {fails:>11,.0f} ({failure_pct:>10.5%})   {total_trials:>11,.0f} ({row_pct:>6.1%})')
        print(f'---------------------------------------------------------------------------------')
        print(f'Total   {t_s_t:>10,.0f} ({t_success_pct:>10.5%})   {t_f_t:>11,.0f} ({t_fail_pct:>10.5%})   {t_t_t:>11,.0f} ({total_pct:>6.1%})')
        
def setup_logging(log_level):
    try:
        if log_level == 'debug':
            logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
        elif log_level == 'info':
            logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
        elif log_level == 'warning':
            logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.WARNING)
        elif log_level == 'error':
            logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.ERROR)
        else:
            raise ValueError(f'\nInvalid logging level: {log_level}')
    except Exception as e:
        logger.error(f'{e}')

def convert_time(time_value, units):
    if units == 'days':
        result = f'{time_value / gbl.elapsed_day:,.2f}' + ' days'
    return result

def main():
    setup_logging(gbl.log_level)
    sim = Simulation()
    sim.run_simulation()
    sim.print_sim_results()

if __name__ == '__main__':
    logger = logging.getLogger()
    main()