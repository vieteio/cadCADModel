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


def wrapPoliciesBlock(function, variableName):
    return lambda params, step, stateHistory, state: {
        variableName: function(
            (params, step, stateHistory),
            [
                state["market customers volume"],
                state["month ad impressions number"],
                state["customers count"],
                state["total token supply"],
                state["token price"],
                state["service quality: successful orders rate, relative executors price"]
            ]
        )
    }

def wrapVariablesBlock(function, variableName):
    return lambda params, step, stateHistory, state, _input: (
        variableName, function(
            (params, step, stateHistory, _input),
            [
                state["market customers volume"],
                state["month ad impressions number"],
                state["customers count"],
                state["total token supply"],
                state["token price"],
                state["service quality: successful orders rate, relative executors price"]
            ]
        )
    )


def total_token_supply_update(parameters_tuple, total_token_supply):
    (parameters, ) = parameters_tuple[:1]
    total_daily_token_reward = parameters["total daily token reward"]
    total_token_supply = total_daily_token_reward + total_token_supply
    return total_token_supply


def conversion_to_payments(customers_count, parameters):
    payments_conversion = parameters["payments conversion"]
    payments_amount = customers_count * payments_conversion
    return payments_amount


def model_service_work_and_update_it_aps_s_quality_metrics(token_price, customers_count, service_quality):
    service_quality = service_quality
    return service_quality


def customer_utility_function(token_price, service_quality, stateHistory):
    successful_orders_rate = service_quality["successful orders rate"]
    relative_executors_price = service_quality["relative executors price"]
    utility = (successful_orders_rate + (2 - relative_executors_price)) / 2
    return utility


def evaluate_action_probabilities(utility):
    probability = evaluate_probability_to_join_DAO(utility)
    probability_to_leave = 1 - probability
    probability_to_continue_to_use_service = probability
    probability_to_recommend_service = probability / 2
    return probability_to_leave, probability_to_continue_to_use_service, probability_to_recommend_service


def update_customers_count_based_on_retention_and_recommendations(probability_to_leave_probability_to_continue_to_use_service_probability_to_recommend_service_tuple, parameters, customers_count, free_market_share):
    (probability_to_leave, probability_to_continue_to_use_service, probability_to_recommend_service, ) = probability_to_leave_probability_to_continue_to_use_service_probability_to_recommend_service_tuple[:3]
    customers_staying_in_service_estimation_number = probability_to_continue_to_use_service * customers_count
    recommendations_conversion = parameters["recommendations conversion"]
    customer_posts_with_recommendation_estimation_number = probability_to_recommend_service * customers_count
    customers_joined_by_recommendation = customer_posts_with_recommendation_estimation_number * recommendations_conversion * free_market_share
    customers_count = customers_staying_in_service_estimation_number + customers_joined_by_recommendation
    return customers_count


def ad_marketing(parameters, market_customers_volume_month_ad_impressions_number_customers_count_token_price_service_quality_col__successful_orders_rate__relative_executors_price_tuple, stateHistory):
    (market_customers_volume, month_ad_impressions_number, customers_count, token_price, service_quality_col__successful_orders_rate__relative_executors_price, ) = market_customers_volume_month_ad_impressions_number_customers_count_token_price_service_quality_col__successful_orders_rate__relative_executors_price_tuple[:5]
    ad_conversion = parameters["ad conversion"]
    free_market_share = (1 - customers_count / market_customers_volume)
    new_customers_familiar_with_ad_number = month_ad_impressions_number * ad_conversion * free_market_share
    new_customers_joined_DAO_estimation_number = DAO_participant_conversion(new_customers_familiar_with_ad_number, token_price, service_quality_col__successful_orders_rate__relative_executors_price, stateHistory)
    return new_customers_joined_DAO_estimation_number


def evaluate_probability_to_join_DAO(utility):
    if utility > 1:
        probability_to_join_to_DAO = 1
        return probability_to_join_to_DAO
    else:
        probability_to_join_to_DAO = utility
        return probability_to_join_to_DAO


def DAO_participant_conversion(new_customers_familiar_with_ad_number, token_price, service_quality_col__successful_orders_rate__relative_executors_price, stateHistory):
    customer_ad_utility_evaluation = customer_utility_function(token_price, service_quality_col__successful_orders_rate__relative_executors_price, stateHistory)
    probability_to_join_to_DAO = evaluate_probability_to_join_DAO(customer_ad_utility_evaluation)
    new_customers_joined_DAO_estimation_number = estimation_evaluation_from_probability(probability_to_join_to_DAO, new_customers_familiar_with_ad_number)
    return new_customers_joined_DAO_estimation_number


def estimation_evaluation_from_probability(probability_to_join_to_DAO, new_customers_familiar_with_ad_number):
    new_customers_joined_DAO_estimation_number = probability_to_join_to_DAO * new_customers_familiar_with_ad_number
    return new_customers_joined_DAO_estimation_number


def total_token_supply_update_2(context, market_customers_volume_month_ad_impressions_number_customers_count_total_token_supply_token_price_service_quality_col__successful_orders_rate__relative_executors_price_tuple):
    (parameters, step, stateHistory, _input, ) = context[:4]
    (market_customers_volume, month_ad_impressions_number, customers_count, total_token_supply, token_price, service_quality_col__successful_orders_rate__relative_executors_price, ) = market_customers_volume_month_ad_impressions_number_customers_count_total_token_supply_token_price_service_quality_col__successful_orders_rate__relative_executors_price_tuple[:6]
    total_token_supply = total_token_supply_update((parameters, ), total_token_supply)
    return total_token_supply


def new_customers_joined_DAO_estimation_number_update(context, variables):
    (parameters, step, stateHistory, ) = context[:3]
    (
        market_customers_volume, 
        month_ad_impressions_number,
        customers_count,
        _,
        token_price,
        service_quality_col__successful_orders_rate__relative_executors_price,
    ) = variables[:6]
    new_customers_joined_DAO_estimation_number = ad_marketing(
        parameters,
        (
            market_customers_volume,
            month_ad_impressions_number,
            customers_count,
            token_price,
            service_quality_col__successful_orders_rate__relative_executors_price,
        ),
        stateHistory
    )
    return new_customers_joined_DAO_estimation_number


def customers_count_update(context, variables):
    (parameters, step, stateHistory, _input, ) = context[:4]
    (_, _, customers_count, _, _, _, ) = variables[:6]
    new_customers_joined_DAO_estimation_number = _input["new customers joined DAO estimation number"]
    customers_count = customers_count + new_customers_joined_DAO_estimation_number
    return customers_count


def token_price_update(context, market_customers_volume_month_ad_impressions_number_customers_count_total_token_supply_token_price_service_quality_col__successful_orders_rate__relative_executors_price_tuple):
    (parameters, step, stateHistory, _input, ) = context[:4]
    (market_customers_volume, month_ad_impressions_number, customers_count, total_token_supply, token_price, service_quality_col__successful_orders_rate__relative_executors_price, ) = market_customers_volume_month_ad_impressions_number_customers_count_total_token_supply_token_price_service_quality_col__successful_orders_rate__relative_executors_price_tuple[:6]
    payments_amount = conversion_to_payments(customers_count, parameters)
    token_price = payments_amount / total_token_supply
    return token_price


def service_quality_col__successful_orders_rate__relative_executors_price_update(context, market_customers_volume_month_ad_impressions_number_customers_count_total_token_supply_token_price_service_quality_col__successful_orders_rate__relative_executors_price_tuple):
    (parameters, step, stateHistory, _input, ) = context[:4]
    (market_customers_volume, month_ad_impressions_number, customers_count, total_token_supply, token_price, service_quality_col__successful_orders_rate__relative_executors_price, ) = market_customers_volume_month_ad_impressions_number_customers_count_total_token_supply_token_price_service_quality_col__successful_orders_rate__relative_executors_price_tuple[:6]
    service_quality_col__successful_orders_rate__relative_executors_price = model_service_work_and_update_it_aps_s_quality_metrics(token_price, customers_count, service_quality_col__successful_orders_rate__relative_executors_price)
    return service_quality_col__successful_orders_rate__relative_executors_price


def customers_count_update_2(context, market_customers_volume_month_ad_impressions_number_customers_count_total_token_supply_token_price_service_quality_col__successful_orders_rate__relative_executors_price_tuple):
    (parameters, step, stateHistory, _input, ) = context[:4]
    (market_customers_volume, month_ad_impressions_number, customers_count, total_token_supply, token_price, service_quality_col__successful_orders_rate__relative_executors_price, ) = market_customers_volume_month_ad_impressions_number_customers_count_total_token_supply_token_price_service_quality_col__successful_orders_rate__relative_executors_price_tuple[:6]
    free_market_share = (1 - customers_count / market_customers_volume)
    customer_service_utility_evaluation = customer_utility_function(token_price, service_quality_col__successful_orders_rate__relative_executors_price, stateHistory)
    (probability_to_leave, probability_to_continue_to_use_service, probability_to_recommend_service) = evaluate_action_probabilities(customer_service_utility_evaluation)
    customers_count = update_customers_count_based_on_retention_and_recommendations((probability_to_leave, probability_to_continue_to_use_service, probability_to_recommend_service, ), parameters, customers_count, free_market_share)
    return customers_count



if __name__ == "__main__":

    state_variables = {
        'market customers volume': 1000000,
        'month ad impressions number': 10000,
        'customers count': 0,
        'total token supply': 0,
        'token price': 0,
        'service quality: successful orders rate, relative executors price': { "successful orders rate": 0.95, "relative executors price": 1 }
    }
    

    params = {
        'ad conversion': [0.1],
        'total daily token reward': [3000],
        'payments conversion': [1000],
        'recommendations conversion': [1.1]
    }
    

    partial_state_update_blocks = [
        {
            'policies': {
                
            },
            'variables': {
                'total token supply': wrapVariablesBlock(total_token_supply_update_2, 'total token supply')
            }
        },
        {
            'policies': {
                'new_customers_joined_DAO_estimation_number_update': wrapPoliciesBlock(new_customers_joined_DAO_estimation_number_update, 'new customers joined DAO estimation number')
            },
            'variables': {
                'customers count': wrapVariablesBlock(customers_count_update, 'customers count')
            }
        },
        {
            'policies': {
                
            },
            'variables': {
                'token price': wrapVariablesBlock(token_price_update, 'token price')
            }
        },
        {
            'policies': {
                
            },
            'variables': {
                'service quality: successful orders rate, relative executors price': wrapVariablesBlock(service_quality_col__successful_orders_rate__relative_executors_price_update, 'service quality: successful orders rate, relative executors price')
            }
        },
        {
            'policies': {
                
            },
            'variables': {
                'customers count': wrapVariablesBlock(customers_count_update_2, 'customers count')
            }
        }
    ]
    
    sim_config = config_sim({
        'T': range(36),
        'N': 10,
        'M': params
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
    
    def extract_dict_column(df, column_name):
        column_names = set()
        # Check if the specified column contains dictionaries with multiple keys
        if df[column_name].apply(lambda x: isinstance(x, dict) and len(x) > 1).any():
            # Extract all unique keys from the dictionaries in the column
            field_names = set()
            for row in df[column_name]:
                if isinstance(row, dict):
                    field_names.update(row.keys())

            # Create new columns for each unique key and initialize them with NaN values
            if len(field_names) > 0:
                for field_name in field_names:
                    column_with_field_name = f"{column_name}: {field_name}"
                    column_names.add(column_with_field_name)
                    df[column_with_field_name] = df[column_name].apply(lambda x: x.get(field_name) if isinstance(x, dict) else None)

            # Drop the original column with dictionaries
            df.drop(columns=[column_name], inplace=True)
        else:
            column_names.add(column_name)

        return df, column_names

    def extract_dict_columns(df, variables: dict):
        variable_column_names = set()
        for column_name in variables.keys():
            df, column_names = extract_dict_column(df, column_name)
            variable_column_names.update(column_names)
        return df, variable_column_names

    def aggregate_runs(df, aggregate_dimension):
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
    
    df, variable_column_names = extract_dict_columns(df, state_variables)
    mean_df, median_df, std_df, min_df = aggregate_runs(df,'timestep')
    

    import matplotlib.pyplot as plt
    
    i = 0
      
    for columnName in variable_column_names:
        plotTitle, yAxisLabel = columnName, ''
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
        if i < len(variable_column_names) - 1:
            plt.figure()
        else:
            plt.show(block=True)
        i += 1
    
