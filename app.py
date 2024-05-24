import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from stoc import stoc

color_p= ["#1984c5", "#63bff0", "#a7d5ed", "#de6e56", "#e14b31", "#c23728"]


# Function definitions
shared_columns = ['idx','dimension', 'rot_type', 'angle', 'mirror', 'wm', 
                  'pair_id', 'obj_id', 'orientation1', 'orientation2', 'image_path_1', 'image_path_2',
                  'marker_id', 'correctAns', 'strategy_response', 'key_resp_strat_control.keys', 'key_resp_strat_control.rt',
                  'vivid_response', 'key_resp_vivid_slider_control.keys', 'key_resp_vivid_slider_control.rt', 'participant']

def get_ans_key(row):
    keys_possible_cols = ['key_resp.keys', 'key_resp_2.keys', 'key_resp_3.keys', 'key_resp_4.keys', 'key_resp_6.keys']
    rt_possible_cols = ['key_resp.rt', 'key_resp_2.rt', 'key_resp_3.rt', 'key_resp_4.rt', 'key_resp_6.rt']
    for key, rt in zip(keys_possible_cols, rt_possible_cols):
        if not pd.isna(row[key]) and row[key] != '':
            return row[key], row[rt]
    return np.nan, np.nan

def get_strategy_response(row):
    if (not pd.isna(row['key_resp_strat_control.keys'])) and (row['key_resp_strat_control.keys'] != 'None') and (row['key_resp_strat_control.keys'] != ''):
        try:    
            strat_resp_list = eval(row['key_resp_strat_control.keys'])
            if len(strat_resp_list) > 0:
                last_key = strat_resp_list[-1]
                if last_key == 'rshift':
                    return 4
                elif last_key == 'slash':
                    return 3
                elif last_key == 'period':
                    return 2
                elif last_key == 'comma':
                    return 1
        except:
            print(row['key_resp_strat_control.keys'])
    return np.nan

def get_vivid_response(row):
    if (not pd.isna(row['key_resp_vivid_slider_control.keys'])) and (row['key_resp_vivid_slider_control.keys'] != 'None') and (row['key_resp_vivid_slider_control.keys'] != ''):
        try:    
            vivid_resp_list = eval(row['key_resp_vivid_slider_control.keys'])
            if len(vivid_resp_list) > 0:
                last_key = vivid_resp_list[-1]
                if last_key == 'rshift':
                    return 4
                elif last_key == 'slash':
                    return 3
                elif last_key == 'period':
                    return 2
                elif last_key == 'comma':
                    return 1
        except:
            print(row['key_resp_vivid_slider_control.keys'])
    return np.nan

def get_block(row):
    if row['dimension'] == '2D':
        if row['wm'] == False:
            return '2D_single'
        elif row['wm'] == True:
            return '2D_wm'
        
    elif row['dimension'] == '3D':
        if row['rot_type'] == 'p':
            if row['wm'] == False:
                return '3Dp_single'
            elif row['wm'] == True:
                return '3Dp_wm'
        elif row['rot_type'] == 'd':
            if row['wm'] == False:
                return '3Dd_single'
            elif row['wm'] == True:
                return '3Dd_wm'

def get_corr(row):
    if row['ans_key'] is np.nan:
        return np.nan
    else:
        if row['correctAns'] == row['ans_key']:
            return 1
        else:
            return 0


def parse_excel(df):
    df_blocks = df[~df['dimension'].isna()]
    df_blocks.reset_index(drop=True, inplace=True)
    df_blocks['idx'] = df_blocks.index
    df_parsed = pd.DataFrame(columns=shared_columns)
    df_parsed['ans_key'] = np.nan
    df_parsed['rt'] = np.nan
    # iterate over the rows of the dataframe to get the ans keys, corr, rt by get_ans_key function
    for idx, row in df_blocks.iterrows():
        key, rt = get_ans_key(row)
        df_parsed.loc[idx, 'ans_key'] = key
        df_parsed.loc[idx, 'rt'] = rt
        for col in shared_columns:
            df_parsed.loc[idx, col] = row[col]
            
        # replace all 'None' values with np.nan
    df_parsed.replace('None', np.nan, inplace=True)
        
    df_parsed['strategy_response'] = df_parsed.apply(get_strategy_response, axis=1)
    df_parsed['vivid_response'] = df_parsed.apply(get_vivid_response, axis=1)

    # fill na values in 'rot_type', 'pair_id', 'orientation1', 'orientation2', 'image_path_2' with not applicable
    for col in ['rot_type', 'pair_id', 'orientation1', 'orientation2', 'image_path_2']:
        df_parsed[col].fillna('na', inplace=True)
        
    df_parsed['block'] = df_parsed.apply(get_block, axis=1)
    df_parsed['corr'] = df_parsed.apply(get_corr, axis=1)
    return df_parsed


# Streamlit app
st.set_page_config(layout="wide")
st.title("Problem solving Data Parsing and Pilot Analysis")

uploaded_file = st.file_uploader("Upload the behavioral data of one participant", type=["csv"])

if uploaded_file:
    toc = stoc()
    df = pd.read_csv(uploaded_file)
    df_parsed = parse_excel(df)
    st.write("Parsed DataFrame:")
    st.dataframe(df_parsed)
    
    # Analysis

    # Average Accuracy
    toc.h2("1. Average Accuracy")

    # Broken down by block
    toc.h3("By Block")
    col1, col2 = st.columns(2)
    accuracy_by_block = df_parsed.groupby('block')['corr'].mean().reset_index().rename(columns={'corr': 'accuracy'})
    with col1:
        st.dataframe(accuracy_by_block)
    with col2:
        # seaborn barplot
        fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
        sns.barplot(data=accuracy_by_block, x='block', y='accuracy', ax=ax, palette=color_p)
        # error bars
        ax.errorbar(x=accuracy_by_block['block'], y=accuracy_by_block['accuracy'], yerr=accuracy_by_block['accuracy'].sem(), fmt='none', ecolor='black', capsize=5)
        # change the labels font size
        ax.set_xlabel('Block', fontsize=14)
        ax.set_ylabel('Accuracy', fontsize=14)
        # remove top and right borders
        sns.despine()
        
        
        st.pyplot(fig)
        

    # Broken down by single vs WM
    toc.h3("By Single vs WM")
    accuracy_by_wm = df_parsed.groupby('wm')['corr'].mean().reset_index().rename(columns={'corr': 'accuracy'})
    accuracy_by_wm['wm'] = accuracy_by_wm['wm'].map({False: 'Single', True: 'WM'})
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(accuracy_by_wm)
    with col2:
        # seaborn barplot
        fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
        sns.barplot(data=accuracy_by_wm, x='wm', y='accuracy', ax=ax,  palette=color_p, width=0.5)
        # error bars
        ax.errorbar(x=accuracy_by_wm['wm'], y=accuracy_by_wm['accuracy'], yerr=accuracy_by_wm['accuracy'].sem(), fmt='none', ecolor='black', capsize=5)
        # change the labels font size
        ax.set_xlabel('WM', fontsize=14)
        ax.set_ylabel('Accuracy', fontsize=14)
        # remove top and right borders
        sns.despine()
        st.pyplot(fig)

    # Broken down by 2D vs 3D
    toc.h3("By 2D vs 3D")
    accuracy_by_dimension = df_parsed.groupby('dimension')['corr'].mean().reset_index().rename(columns={'corr': 'accuracy'})
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(accuracy_by_dimension)
    with col2:
        # seaborn barplot
        fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
        sns.barplot(data=accuracy_by_dimension, x='dimension', y='accuracy', ax=ax, palette=color_p, width=0.5)
        # error bars
        ax.errorbar(x=accuracy_by_dimension['dimension'], y=accuracy_by_dimension['accuracy'], yerr=accuracy_by_dimension['accuracy'].sem(), fmt='none', ecolor='black', capsize=5)
        # change the labels font size
        ax.set_xlabel('Dimension', fontsize=14)
        ax.set_ylabel('Accuracy', fontsize=14)
        # remove top and right borders
        sns.despine()
        st.pyplot(fig)
        

    # Effect of angular difference within blocks
    toc.h3("Angular Difference Within Blocks")
    angular_effect = df_parsed.groupby(['block', 'angle'])['corr'].mean().reset_index().rename(columns={'corr': 'accuracy'})
    
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(angular_effect)
    with col2:
        # seaborn barplot
        fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
        sns.barplot(data=angular_effect, x='block', y='accuracy', hue='angle', ax=ax, palette=color_p)
        plt.legend(title='Angle', loc='upper left', bbox_to_anchor=(1, 1))
        # change the labels font size
        ax.set_xlabel('Angle', fontsize=14)
        ax.set_ylabel('Accuracy', fontsize=14)
        # remove top and right borders
        sns.despine()
        st.pyplot(fig)
        
    # Average Reaction Time
    toc.h2("2. Average Reaction Time")
    
    ## whether it will include the correct answer or not
    toc.h3("By Correct vs Incorrect")
    rt_by_corr = df_parsed.groupby('corr')['rt'].agg(['mean', 'std']).reset_index()
    
    # t-test
    corr_rt = df_parsed[df_parsed['corr'] == 1]['rt'].dropna()
    incorr_rt = df_parsed[df_parsed['corr'] == 0]['rt'].dropna()
    # independent t-test
    t, p = ttest_ind(corr_rt, incorr_rt)
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(rt_by_corr)
    with col2:
        # boxplot
        fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
        sns.boxplot(data=df_parsed, x='corr', y='rt', ax=ax, palette=color_p)
        plt.title(f"Boxplot, t-test: t={t:.2f}, p={p:.2f}")
        # change the labels font size
        ax.set_xlabel('Correct', fontsize=14)
        ax.set_ylabel('Reaction Time', fontsize=14)
        # remove top and right borders
        sns.despine()
        st.pyplot(fig)
    
    # Broken down by block
    toc.h3("By Block")
    rt_by_block = df_parsed.groupby('block')['rt'].mean().reset_index().rename(columns={'rt': 'rt_mean'})
    
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(rt_by_block)
    with col2:
        # seaborn barplot
        fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
        sns.barplot(data=rt_by_block, x='block', y='rt_mean', ax=ax, palette=color_p, ci='sd')
        # error bars
        ax.errorbar(x=rt_by_block['block'], y=rt_by_block['rt_mean'], yerr=df_parsed.groupby('block')['rt'].sem(), fmt='none', ecolor='black', capsize=5)
        # change the labels font size
        ax.set_xlabel('Block', fontsize=14)
        ax.set_ylabel('Reaction Time', fontsize=14)
        # remove top and right borders
        sns.despine()
        st.pyplot(fig)
        
    # Broken down by single vs WM
    toc.h3("By Single vs WM")
    rt_by_wm = df_parsed.groupby('wm')['rt'].mean().reset_index().rename(columns={'rt': 'rt_mean'})
    rt_by_wm['wm'] = rt_by_wm['wm'].map({False: 'Single', True: 'WM'})
    
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(rt_by_wm)
    with col2:
        # seaborn barplot
        fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
        sns.barplot(data=rt_by_wm, x='wm', y='rt_mean', ax=ax, palette=color_p, width=0.5)
        # error bars
        ax.errorbar(x=rt_by_wm['wm'], y=rt_by_wm['rt_mean'], yerr=df_parsed.groupby('wm')['rt'].sem(), fmt='none', ecolor='black', capsize=5)
        # change the labels font size
        ax.set_xlabel('WM', fontsize=14)
        ax.set_ylabel('Reaction Time', fontsize=14)
        # remove top and right borders
        sns.despine()
        st.pyplot(fig)
        
    # Broken down by 2D vs 3D
    toc.h3("By 2D vs 3D")
    rt_by_dimension = df_parsed.groupby('dimension')['rt'].mean().reset_index().rename(columns={'rt': 'rt_mean'})
    
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(rt_by_dimension)
    with col2:
        # seaborn barplot
        fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
        sns.barplot(data=rt_by_dimension, x='dimension', y='rt_mean', ax=ax, palette=color_p, width=0.5)
        # error bars
        ax.errorbar(x=rt_by_dimension['dimension'], y=rt_by_dimension['rt_mean'], yerr=df_parsed.groupby('dimension')['rt'].sem(), fmt='none', ecolor='black', capsize=5)
        # change the labels font size
        ax.set_xlabel('Dimension', fontsize=14)
        ax.set_ylabel('Reaction Time', fontsize=14)
        # remove top and right borders
        sns.despine()
        st.pyplot(fig)
    
    # Broken down by angular difference within blocks
    toc.h3("Angular Difference Within Blocks")
    rt_by_angle = df_parsed.groupby(['block', 'angle'])['rt'].mean().reset_index().rename(columns={'rt': 'rt_mean'})
    
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(rt_by_angle)
    with col2:
        # seaborn barplot
        fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
        sns.barplot(data=rt_by_angle, x='block', y='rt_mean', hue='angle', ax=ax, palette=color_p)
        plt.legend(title='Angle', loc='upper left', bbox_to_anchor=(1, 1))
        # change the labels font size
        ax.set_xlabel('Angle', fontsize=14)
        ax.set_ylabel('Reaction Time', fontsize=14)
        # remove top and right borders
        sns.despine()
        st.pyplot(fig)
    
    # Performance over time How does accuracy and reaction time change over time (across blocks sequentially)? Are participants getting better or faster as they progress through the experiment?
    toc.h2("3. Performance Over Time")
    
    col1, col2 = st.columns(2)
    with col1:  
        # Accuracy
        toc.h3("Accuracy")
        # running average accuracy over idx 
        df_parsed['running_avg_acc'] = df_parsed['corr'].expanding().mean()
        fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
        sns.lineplot(data=df_parsed, x='idx', y='running_avg_acc', ax=ax, color=color_p[0])
        # color background for each block
        for idx, block in enumerate(df_parsed['block'].unique()):
            block_idx = df_parsed[df_parsed['block'] == block]['idx']
            ax.axvspan(block_idx.min(), block_idx.max(), alpha=0.1, color=color_p[idx])
            # add block label in the bottom
            ax.text(block_idx.mean(), 0.015, block, ha='center', va='center', fontsize=8, color='black')
            
        # change the labels font size
        ax.set_xlabel('Index', fontsize=14)
        ax.set_ylabel('Running Average Accuracy', fontsize=14)
        # remove top and right borders
        sns.despine()
        st.pyplot(fig)
    
    with col2:
        # Reaction Time
        toc.h3("Reaction Time")
        # running average reaction time over idx
        df_parsed['running_avg_rt'] = df_parsed['rt'].expanding().mean()
        fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
        sns.lineplot(data=df_parsed, x='idx', y='running_avg_rt', ax=ax, color=color_p[0])
        # color background for each block
        for idx, block in enumerate(df_parsed['block'].unique()):
            block_idx = df_parsed[df_parsed['block'] == block]['idx']
            ax.axvspan(block_idx.min(), block_idx.max(), alpha=0.1, color=color_p[idx])
            # add block label in the bottom
            ax.text(block_idx.mean(), 1.3, block, ha='center', va='center', fontsize=8, color='black')
        # change the labels font size
        ax.set_xlabel('Index', fontsize=14)
        ax.set_ylabel('Running Average Reaction Time', fontsize=14)
        # remove top and right borders
        sns.despine()
        st.pyplot(fig)
    
    # strategy with performance
    toc.h2("4. Strategy with Performance")
    
    st.write("Strategy Response count")
    strategy_cnt = df_parsed['strategy_response'].value_counts().reset_index()
    st.dataframe(strategy_cnt)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Accuracy
        strategy_acc = df_parsed.groupby('strategy_response')['corr'].agg([ 'mean', 'std']).reset_index().rename(columns={'mean': 'accuracy'})
        # strategy response should range from 1 to 4, participants can only have part of the strategy
        strategy_acc['strategy_response'] = strategy_acc['strategy_response'].astype(int)
        strategy_acc_full = pd.DataFrame({'strategy_response': [1, 2, 3, 4]})
        strategy_acc = pd.merge(strategy_acc_full, strategy_acc, on='strategy_response', how='left')
        toc.h3("Accuracy")
        st.dataframe(strategy_acc)
        
        fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
        sns.barplot(x='strategy_response', y='accuracy', data=strategy_acc, ax=ax, palette=color_p)
        plt.ylim(0, 1)
        # Add "X" markers for missing values
        for i, acc in enumerate(strategy_acc['accuracy']):
            if pd.isna(acc):
                x_coord = i
                y_coord = 0.5
                plt.text(x_coord, y_coord, 'X', ha='center', va='center', fontsize=14, color='black')

        # Set labels and title
        plt.xlabel('Strategy Response')
        plt.ylabel('Accuracy')
        st.pyplot(fig)
        
    with col2:
        # Reaction Time
        strategy_rt = df_parsed.groupby('strategy_response')['rt'].agg([ 'mean', 'std']).reset_index().rename(columns={'mean': 'rt_mean'})
        strategy_rt['strategy_response'] = strategy_rt['strategy_response'].astype(int)
        strategy_rt_full = pd.DataFrame({'strategy_response': [1, 2, 3, 4]})
        strategy_rt = pd.merge(strategy_rt_full, strategy_rt, on='strategy_response', how='left')
        toc.h3("Reaction Time")
        st.dataframe(strategy_rt)
        
        fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
        sns.barplot(x='strategy_response', y='rt_mean', data=strategy_rt, ax=ax, palette=color_p)
        # Add "X" markers for missing values
        for i, rt in enumerate(strategy_rt['rt_mean']):
            if pd.isna(rt):
                x_coord = i
                y_coord = 1
                plt.text(x_coord, y_coord, 'X', ha='center', va='center', fontsize=14, color='black')

        # Set labels and title
        plt.xlabel('Strategy Response')
        plt.ylabel('Reaction Time')
        st.pyplot(fig)
        
        
    # More analysis on your choice
    toc.h2("5. More Analysis by Yourself.")
    st.write("If you want to do more analysis, you can select the columns to group by, columns to aggregate and the aggregate function.")
    st.write("By default it's showing the accuracy grouped by block.")
    # Sidebar for user input
   
    groupby_cols = st.multiselect('Group by', ['block', 'dimension', 'wm', 'angle', 'rot_type', 'obj_id'], default=['block'])
    agg_cols = st.multiselect('Column to aggregate', ['corr', 'rt', 'strategy_response', 'vivid_response'], default=['corr'])
    aggregate_func = st.multiselect('Aggregate Function', ['mean', 'count', 'max', 'min', 'std'], default=['mean'])

    # Main content
    if groupby_cols and aggregate_func and agg_cols:
        with st.expander('Grouped Data'):
            groupby_data = df_parsed.groupby(groupby_cols)[agg_cols].agg(aggregate_func)
            # flatten the multiindex columns
            groupby_data.columns = ['_'.join(col).strip() for col in groupby_data.columns.values]
            st.dataframe(groupby_data)

    toc.toc()

