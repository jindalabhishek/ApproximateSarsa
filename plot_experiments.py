import os

import pandas as pd
import matplotlib.pyplot as plt
import shutil
from scipy import stats
import seaborn as sns


def test_func():
    q_dfs = []
    for file in os.listdir('ApproximateQAgent'):
        df = pd.read_csv('ApproximateQAgent/' + file)
        q_dfs.append(df)

    q_df_concat = pd.concat(q_dfs)
    by_row_index = q_df_concat.groupby(q_df_concat.index)
    df_means = by_row_index.mean()

    q_averaged_run_data = sum(q_dfs) / len(q_dfs)
    q_averaged_run_data = q_averaged_run_data.rolling(100).mean()

    s_dfs = []
    for file in os.listdir('ApproximateSarsaAgent'):
        df = pd.read_csv('ApproximateSarsaAgent/' + file)
        s_dfs.append(df)

    s_df_concat = pd.concat(s_dfs)
    by_row_index = s_df_concat.groupby(s_df_concat.index)
    df_means = by_row_index.mean()

    s_averaged_run_data = sum(s_dfs) / len(s_dfs)
    s_averaged_run_data = s_averaged_run_data.rolling(100).mean()

    # averaged_run_data = pd.concat([q_averaged_run_data, s_averaged_run_data])
    # averaged_run_data.plot()
    ax = q_averaged_run_data.plot()
    s_averaged_run_data.plot(ax=ax)
    ax.legend(['Approximate Q Agent', 'True Online Sarsa(l=0) LFA Agent'])
    plt.xlabel("Number of Episodes")
    plt.ylabel("Total Score averaged over 100 runs")

    plt.show()
    print(len(q_averaged_run_data))
    last_index = q_averaged_run_data.index[len(q_averaged_run_data) - 1]
    val = q_averaged_run_data.loc[last_index][0]
    min_val = 0.95 * val
    max_val = 1.05 * val
    for idx in reversed(q_averaged_run_data.index):
        if min_val <= q_averaged_run_data.loc[idx][0] <= max_val:
            continue
        else:
            print(idx)
            print(q_averaged_run_data.loc[idx])
            break

    print(len(s_averaged_run_data))
    last_index = s_averaged_run_data.index[len(s_averaged_run_data) - 1]
    val = s_averaged_run_data.loc[last_index][0]
    min_val = 0.95 * val
    max_val = 1.05 * val
    for idx in reversed(s_averaged_run_data.index):
        if min_val <= s_averaged_run_data.loc[idx][0] <= max_val:
            continue
        else:
            print(idx)
            print(s_averaged_run_data.loc[idx])
            break


def get_convergence_episode(folder_name):
    convergence_episodes = []
    for file in os.listdir(folder_name):
        df = pd.read_csv(folder_name + '/' + file).rolling(100).mean()
        # df.plot()
        last_index = df.index[len(df) - 1]
        val = df.loc[last_index][0]
        # print('Last: ')
        # print(val)
        min_val = 0.80 * val
        max_val = 1.2 * val
        for idx in reversed(df.index):
            if min_val <= df.loc[idx][0] <= max_val:
                continue
            else:
                # print('Episode: ' + str(idx + 1))
                # print('Reward: ')
                # print(df.loc[idx + 1])
                convergence_episodes.append(idx + 1)
                break
    # plt.show()
    return convergence_episodes


def get_average_convergence_speed(folder_name):
    average_rewards = []
    for file in os.listdir(folder_name):
        df = pd.read_csv(folder_name + '/' + file).rolling(100).mean()
        val = df.mean()
        average_rewards.append(val)
    return average_rewards


def plot_reward_for_attribute(folder_names, legends, num_runs, num_episodes, rolling_average):
    consolidated_dfs = []
    for folder_name in folder_names:
        dfs = []
        for file in os.listdir(folder_name):
            df = pd.read_csv(folder_name + '/' + file)
            dfs.append(df)
        averaged_df = sum(dfs) / len(dfs)
        consolidated_dfs.append(averaged_df.rolling(rolling_average).mean())
    ax = consolidated_dfs[0].plot()
    for i in range(1, len(consolidated_dfs)):
        consolidated_dfs[i].plot(ax=ax)
    ax.legend(legends)
    plt.xlabel("Number of Episodes")
    plt.ylabel("Total Score averaged over " + str(num_runs) + " runs")
    plt.show()


# q_episodes = get_convergence_episode('ApproximateQAgent')
# s9_episodes = get_convergence_episode('ApproximateSarsaAgent_lambda9_convergence')
# s_episodes = get_convergence_episode('ApproximateSarsaAgent')

# print(q_episodes)
# print(s9_episodes)
# print(s_episodes)
#
# print(sum(q_episodes)/len(q_episodes))
# print(sum(s9_episodes)/len(s9_episodes))
# print(sum(s_episodes)/len(s_episodes))

# for file in os.listdir(dir_path):
#     num = int(file.replace('test-', '').replace('.csv', ''))
#     mod = num%6-1
#     file_name = new_path + layouts[mod] + '/' + file
#     print(file_name)
#     os.makedirs(new_path+layouts[mod], exist_ok=True)
#     shutil.move(dir_path + '/' + file, file_name)

def unpaired_t_test(df1, df2):
    statistic, p_value = stats.ttest_ind(df1, df2)
    print(statistic, p_value)


def paired_t_test(df1, df2):
    statistic, p_value = stats.ttest_rel(df1, df2)
    print(statistic, p_value)


def main():
    plot_reward_for_attribute(
        ['ApproximateSarsaAgentSimpleExtractor', 'ApproximateQAgentSimpleExtractor'],
        ['ApproximateSarsaAgentSimpleExtractor', 'ApproximateQAgentSimpleExtractor'], 14, 1000, 100)

    plot_reward_for_attribute(
        ['ApproximateSarsaAgent_lambda5', 'ApproximateSarsaAgent_lambda75', 'ApproximateSarsaAgent_lambda9'],
        ['lambda=0.5', 'lambda=0.75', 'lambda=0.9'], 50, 200, 20)

    plot_reward_for_attribute(
        ['ApproximateQAgent', 'ApproximateSarsaAgent', 'ApproximateSarsaAgent_lambda9_convergence'],
        ['Approximate Q Agent', 'True Online Sarsa(l=0) LFA Agent', 'True Online Sarsa(l=0.9) LFA Agent'],
        50, 1000, 10)

    layouts = ['trickyClassic', 'powerClassic', 'capsuleClassic', 'originalClassic', 'mediumGrid', 'smallGrid']
    feat_extractors = ['IdentityExtractor', 'SimpleExtractor', 'ComplexExtractor', 'IdentityExtractor',
                       'ComplexExtractor', 'ComplexExtractor']
    ghostNum = [2, 3, 4, 4, 3, 3]
    # dir_path = 'ApproximateSarsaAgent_mult_run'
    new_path = 'run_layouts/ApproximateSarsaAgent_'
    new_path_layouts = []
    legends = []

    for i in range(0, len(layouts)):
        new_path_layouts.append(new_path + layouts[i])
        legends.append('_'.join([layouts[i], feat_extractors[i], str(ghostNum[i])]))

    plot_reward_for_attribute(new_path_layouts, legends, 4, 1000, 10)

    q_convergence_speed = get_average_convergence_speed('ApproximateQAgent')
    s9_convergence_speed = get_average_convergence_speed('ApproximateSarsaAgent_lambda9_convergence')
    s_convergence_speed = get_average_convergence_speed('ApproximateSarsaAgent')

    print(sum(q_convergence_speed) / len(q_convergence_speed))
    print(sum(s9_convergence_speed) / len(s9_convergence_speed))
    print(sum(s_convergence_speed) / len(s_convergence_speed))

    unpaired_t_test(q_convergence_speed, s9_convergence_speed)
    unpaired_t_test(q_convergence_speed, s_convergence_speed)
    paired_t_test(s9_convergence_speed, s_convergence_speed)

    layout_dfs = []
    for i in range(0, len(new_path_layouts)):
        path = new_path_layouts[i]
        layout_convergence_speed = get_average_convergence_speed(path)
        df = pd.DataFrame(layout_convergence_speed)
        df = df.rename(columns={'Score': layouts[i]})
        print(df.head())
        layout_dfs.append(df)

    pd_layout_df = pd.concat(layout_dfs)
    ax = sns.boxplot(data=pd_layout_df)
    ax = sns.swarmplot(data=pd_layout_df)
    plt.xlabel('Environments with different configuration (Layout_Extractor_GhostNumber)')
    plt.ylabel('Average Reward for 1000 episodes rolled mean over 100 episodes (4 runs)')
    plt.show()
    fvalue, pvalue = stats.f_oneway(*layout_dfs)
    print(fvalue, pvalue)


main()
