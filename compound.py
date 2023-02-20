# %%
from cadCAD.configuration import Experiment
from cadCAD.configuration.utils import bound_norm_random, ep_time_step, config_sim, access_block

import numpy as np
import pandas as pd
from cadCAD.engine import ExecutionMode, ExecutionContext,Executor

def run(configs):
    '''
    Definition:
    Run simulation
    '''
    exec_mode = ExecutionMode()
    local_mode_ctx = ExecutionContext(context=exec_mode.local_mode)

    simulation = Executor(exec_context=local_mode_ctx, configs=configs)
    raw_system_events, tensor_field, sessions = simulation.execute()
    # Result System Events DataFrame
    df = pd.DataFrame(raw_system_events)
    return df

if __name__ == "__main__":

    params = {
        'eta': [.33], # for payments_volume_generator
        'tampw': [3000000], # payments volume limit
        'COMP_reward': [2880], # dayly COMP supply
    }

    # Initial States
    initial_values = {
                'Payments Volume': float(100), #unit: fiat
                'COMP supply': float(0), #unit: tokens count
                'Token Price': float(0),#unit: fiat
    }

    state_variables = {
                'Payments Volume': initial_values['Payments Volume'], #unit: fiat
                'COMP supply': initial_values['COMP supply'], #unit: tokens count
                'Token Price': initial_values['Token Price'], #unit: fiat
    }

    def payments_volume_generator(params, step, sL, s, _input):
        y = 'Payments Volume'
        x = s['Payments Volume'] * (1 + 2 * params['eta'] * np.random.rand() * (1 - s['Payments Volume'] / params['tampw']))
        return (y, x)

    def COMP_supply_generator(params, step, sL, s, _input):
        y = 'COMP supply'
        x = s['COMP supply'] + params['COMP_reward']
        return (y, x)


    def update_token_price(params, step, sL, s, _input):
        y = 'Token Price'
        x = s['Payments Volume'] / s['COMP supply']
        return (y, x)

    partial_state_update_blocks = [
        {
            'policies':
            {
            },
            'variables':
            {
            }
        },
        {
            'policies':
            {
            },
            'variables':
            {
                'Payments Volume': payments_volume_generator,
                'COMP supply': COMP_supply_generator,
            }
        },
        {
            'policies':
            {
            },
            'variables':
            {
                'Token Price': update_token_price
            }
        },
    ]

    sim_config = config_sim({
        'T': range(46), #day 
        'N': 10,
        'M': params,
    })

    seeds = {
        'a': np.random.RandomState(2),
    }

    exp = Experiment()

    exp.append_configs(
        sim_configs=sim_config,
        initial_state=state_variables,
        seeds=seeds,
        partial_state_update_blocks=partial_state_update_blocks
    )

    df = run(exp.configs)


    # %%
    def aggregate_runs(df,aggregate_dimension):
        '''
        Function to aggregate the monte carlo runs along a single dimension.
        Parameters:
        df: dataframe name
        aggregate_dimension: the dimension you would like to aggregate on, the standard one is timestep.
        Example run:
        mean_df,median_df,std_df,min_df = aggregate_runs(df,'timestep')
        '''
        aggregate_dimension = aggregate_dimension

        mean_df = df.groupby(aggregate_dimension).mean().reset_index()
        median_df = df.groupby(aggregate_dimension).median().reset_index()
        std_df = df.groupby(aggregate_dimension).std().reset_index()
        min_df = df.groupby(aggregate_dimension).min().reset_index()

        return mean_df, median_df, std_df, min_df

    mean_df, median_df, std_df, min_df = aggregate_runs(df,'timestep')

    # %%
    import matplotlib.pyplot as plt

    plotLables = {
        'Payments Volume': ('Payments Volume', 'volume'),
        'COMP supply': ('COMP supply', 'volume'),
        'Token Price': ('Token Price', '$')
    }

    i = 0
    for columnName in state_variables.keys():
        plotTitle, yAxisLabel = plotLables[columnName]
        coefficients, residuals, _, _, _ = np.polyfit(
            range(len(mean_df[columnName])),
            mean_df[columnName],
            1,
            full=True
        )
        plt.plot(mean_df[columnName], label=columnName)
        # plt.plot([coefficients[0] * x + coefficients[1] for x in range(len(mean_df['Payments Volume']))], label='trend line')
        plt.title(columnName)
        plt.xlabel('time step')
        plt.ylabel('value')
        plt.ylim(0)
        plt.legend()
        if i < 2:
            plt.figure()
        else:
            plt.show(block=True)
        i += 1

    # COMP_price_ratio = mean_df['Token Price'].max() / mean_df['COMP supply'].max()
    # plt.plot([mean_df['COMP supply'][x] * COMP_price_ratio for x in range(len(mean_df['COMP supply']))],label='COMP supply')
