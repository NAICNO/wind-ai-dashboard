#!/bin/env python3
#sabryr Norwegian Ai cloud
#skelliton code generated with deepseek

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import numpy as np
from matplotlib.gridspec import GridSpec

def load_and_validate_data(yesterday_file, competition_file):
    """
    Load and validate data from both CSV files
    """
    try:
        # Load data
        df_yesterday = pd.read_csv(yesterday_file)
        df_competition = pd.read_csv(competition_file)
        
        print(f"Successfully loaded data:")
        print(f"  Yesterday file: {yesterday_file} - {len(df_yesterday)} teams")
        print(f"  Competition file: {competition_file} - {len(df_competition)} teams")
        
        # Validate columns
        required_columns = ['id', 'team', 'rmse']
        for df, name in [(df_yesterday, "Yesterday"), (df_competition, "Competition")]:
            if not all(col in df.columns for col in required_columns):
                print(f"Error: {name} file must contain columns: {required_columns}")
                print(f"Found columns: {df.columns.tolist()}")
                return None, None
        
        # Check for exactly one entry per team
        for df, name in [(df_yesterday, "Yesterday"), (df_competition, "Competition")]:
            team_counts = df['team'].value_counts()
            if not all(team_counts == 1):
                duplicate_teams = team_counts[team_counts > 1].index.tolist()
                print(f"Error: {name} file has multiple entries for teams: {duplicate_teams}")
                return None, None
        
        # Check if teams match between files
        yesterday_teams = set(df_yesterday['team'].unique())
        competition_teams = set(df_competition['team'].unique())
        
        if yesterday_teams != competition_teams:
            missing_in_comp = yesterday_teams - competition_teams
            missing_in_yest = competition_teams - yesterday_teams
            
            if missing_in_comp:
                print(f"Warning: Teams in yesterday file but not in competition: {missing_in_comp}")
            if missing_in_yest:
                print(f"Warning: Teams in competition file but not in yesterday: {missing_in_yest}")
        
        return df_yesterday, df_competition
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return None, None
    except Exception as e:
        print(f"Error reading CSV files: {e}")
        return None, None

def create_summary_table(ax, df, title, team_order=None):
    """
    Create a summary table for a dataframe with rank column
    """
    # Create a copy to avoid modifying original
    df_table = df.copy()
    
    # Add rank column (lower RMSE = better rank = lower number)
    df_table['rank'] = df_table['rmse'].rank(method='min').astype(int)
    
    if team_order is not None:
        # Reorder dataframe according to provided team order
        df_table = df_table.set_index('team').loc[team_order].reset_index()
    else:
        # Sort by rank (ascending) if no specific order provided
        df_table = df_table.sort_values('rank')
    
    # Prepare table data
    table_data = []
    for _, row in df_table.iterrows():
        table_data.append([row['rank'], row['team'], f"{row['rmse']:.6f}", row['id']])
    
    # Create table
    table = ax.table(cellText=table_data,
                    colLabels=['Rank', 'Team', 'RMSE', 'ID'],
                    loc='center',
                    cellLoc='center',
                    colWidths=[0.15, 0.3, 0.3, 0.15])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#4F81BD')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style data rows and rank column
    for i in range(1, len(table_data) + 1):
        for j in range(4):
            if i % 2 == 0:  # Zebra striping
                table[(i, j)].set_facecolor('#F2F2F2')
            
            # Highlight top 3 ranks
            if j == 0 and i <= 3:  # Rank column, top 3 rows
                if i == 1:  # Rank 1 - Gold
                    table[(i, j)].set_facecolor('#FFD700')
                elif i == 2:  # Rank 2 - Silver
                    table[(i, j)].set_facecolor('#C0C0C0')
                elif i == 3:  # Rank 3 - Bronze
                    table[(i, j)].set_facecolor('#CD7F32')
    
    ax.set_title(title, fontweight='bold', fontsize=10, pad=10)
    ax.axis('off')
    
    return df_table

def generate_comparison_plot(df_yesterday, df_competition, output_file='rmse_comparison.png'):
    """
    Generate comparison plot with summary tables at the top including ranks
    """
    # Merge data for comparison
    df_merged = pd.merge(df_yesterday, df_competition, on='team', 
                         suffixes=('_yesterday', '_competition'))
    
    # Calculate differences
    df_merged['rmse_diff'] = df_merged['rmse_competition'] - df_merged['rmse_yesterday']
    df_merged['improvement'] = df_merged['rmse_diff'] < 0
    
    # Sort by competition RMSE (best first) for consistent ordering
    df_merged = df_merged.sort_values('rmse_competition')
    team_order = df_merged['team'].tolist()
    
    # Create figure with GridSpec for better layout control
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1.5])
    
    # Create subplots
    ax_table1 = fig.add_subplot(gs[0, 0])  # Yesterday table
    ax_table2 = fig.add_subplot(gs[0, 1])  # Competition table
    ax1 = fig.add_subplot(gs[1, 0])        # Bar chart
    ax2 = fig.add_subplot(gs[1, 1])        # Scatter plot
    ax3 = fig.add_subplot(gs[2, 0])        # Improvement chart
    ax4 = fig.add_subplot(gs[2, 1])        # Ranking chart
    
    # Set main title
    fig.suptitle('RMSE Analysis: Yesterday vs Competition Period (with Rankings)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Create summary tables with ranks
    df_yesterday_table = create_summary_table(ax_table1, df_yesterday, 
                                             'Yesterday Performance (Ranked by RMSE)')
    df_competition_table = create_summary_table(ax_table2, df_competition, 
                                               'Competition Performance (Ranked by RMSE)')
    
    # Add rank columns to merged dataframe for plots
    df_merged['rank_yesterday'] = df_merged['rmse_yesterday'].rank(method='min').astype(int)
    df_merged['rank_competition'] = df_merged['rmse_competition'].rank(method='min').astype(int)
    df_merged['rank_change'] = df_merged['rank_yesterday'] - df_merged['rank_competition']
    
    # Plot 1: Side-by-side bar chart with ranks annotated
    x_pos = np.arange(len(df_merged))
    width = 0.35
    
    bars1 = ax1.bar(x_pos - width/2, df_merged['rmse_yesterday'], width, 
                   label='Yesterday', alpha=0.7, color='skyblue')
    bars2 = ax1.bar(x_pos + width/2, df_merged['rmse_competition'], width, 
                   label='Competition', alpha=0.7, color='lightcoral')
    
    ax1.set_xlabel('Team')
    ax1.set_ylabel('RMSE')
    ax1.set_title('RMSE Comparison: Yesterday vs Competition (with Ranks)', fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(df_merged['team'], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels and ranks on bars
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        # RMSE values
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        
        # Rank annotations
        rank_yest = df_merged.iloc[i]['rank_yesterday']
        rank_comp = df_merged.iloc[i]['rank_competition']
        
        ax1.text(bar1.get_x() + bar1.get_width()/2., height1 + 0.001,
                f'{height1:.4f}\n(R#{rank_yest})', ha='center', va='bottom', fontsize=6)
        ax1.text(bar2.get_x() + bar2.get_width()/2., height2 + 0.001,
                f'{height2:.4f}\n(R#{rank_comp})', ha='center', va='bottom', fontsize=6)
    
    # Plot 2: Scatter plot showing relationship with ranks
    scatter = ax2.scatter(df_merged['rmse_yesterday'], df_merged['rmse_competition'], 
                         s=60, alpha=0.7, color='green', edgecolors='black', linewidth=0.5)
    
    # Add team labels with ranks to points
    for i, row in df_merged.iterrows():
        ax2.annotate(f"{row['team']}\n(Y#{row['rank_yesterday']}/C#{row['rank_competition']})", 
                    (row['rmse_yesterday'], row['rmse_competition']),
                    xytext=(5, 5), textcoords='offset points', fontsize=7,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
    
    # Add reference line (y = x)
    min_val = min(df_merged[['rmse_yesterday', 'rmse_competition']].min())
    max_val = max(df_merged[['rmse_yesterday', 'rmse_competition']].max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='y = x')
    
    ax2.set_xlabel('Yesterday RMSE')
    ax2.set_ylabel('Competition RMSE')
    ax2.set_title('RMSE Correlation with Rank Annotations\n(Y#=Yesterday Rank, C#=Competition Rank)', 
                  fontweight='bold', fontsize=10)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Improvement/change plot with rank change
    colors = ['green' if imp else 'red' for imp in df_merged['improvement']]
    bars = ax3.bar(range(len(df_merged)), df_merged['rmse_diff'], color=colors, alpha=0.7, edgecolor='black')
    
    ax3.set_xlabel('Team')
    ax3.set_ylabel('RMSE Difference (Competition - Yesterday)')
    ax3.set_title('RMSE Change: Negative = Improvement\n(with Rank Change)', fontweight='bold')
    ax3.set_xticks(range(len(df_merged)))
    ax3.set_xticklabels(df_merged['team'], rotation=45, ha='right')
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
    ax3.grid(True, alpha=0.3)
    
    # Add value labels and rank change on bars
    for i, (bar, value) in enumerate(zip(bars, df_merged['rmse_diff'])):
        height = bar.get_height()
        rank_change = df_merged.iloc[i]['rank_change']
        
        va = 'bottom' if height >= 0 else 'top'
        color = 'black' if abs(height) > 0.01 else 'white'
        y_offset = 0.001 if height >= 0 else -0.001
        
        # Rank change symbol
        rank_symbol = "↑" if rank_change > 0 else "↓" if rank_change < 0 else "→"
        rank_color = "green" if rank_change > 0 else "red" if rank_change < 0 else "gray"
        
        ax3.text(bar.get_x() + bar.get_width()/2., height + y_offset, 
                f'{value:+.4f}\n{rank_symbol}{abs(rank_change)}', 
                ha='center', va=va, fontsize=7, color=color, fontweight='bold')
    
    # Plot 4: Ranking comparison with detailed annotations
    ax4.scatter(df_merged['rank_yesterday'], df_merged['rank_competition'], 
               s=80, alpha=0.7, color='purple', edgecolors='black', linewidth=0.5)
    
    # Add team labels with detailed rank info
    for i, row in df_merged.iterrows():
        rank_change = row['rank_change']
        change_symbol = "↑" if rank_change > 0 else "↓" if rank_change < 0 else "→"
        change_color = "green" if rank_change > 0 else "red" if rank_change < 0 else "gray"
        
        ax4.annotate(f"{row['team']}\n{change_symbol}{abs(rank_change)}", 
                    (row['rank_yesterday'], row['rank_competition']),
                    xytext=(8, 8), textcoords='offset points', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                    color=change_color, fontweight='bold')
    
    ax4.set_xlabel('Yesterday Rank (1 = Best)')
    ax4.set_ylabel('Competition Rank (1 = Best)')
    ax4.set_title('Rank Comparison: ↑ = Improved Rank, ↓ = Declined Rank', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Add reference line and improvement regions
    max_rank = max(df_merged[['rank_yesterday', 'rank_competition']].max())
    ax4.plot([1, max_rank], [1, max_rank], 'r--', alpha=0.5, label='No change')
    
    # Add improvement region shading
    ax4.fill_between([1, max_rank], [1, 1], [max_rank, max_rank], 
                    alpha=0.1, color='green', label='Improvement area')
    
    ax4.legend()
    ax4.invert_xaxis()  # Reverse axes so rank 1 is top-left
    ax4.invert_yaxis()
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    
    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nComparison plot with rankings saved as: {output_file}")
    
    return df_merged

def print_statistics(df_merged):
    """
    Print comprehensive statistics about the comparison
    """
    print(f"\n{'='*60}")
    print("COMPREHENSIVE STATISTICS")
    print(f"{'='*60}")
    
    print(f"\nDataset Overview:")
    print(f"Total teams compared: {len(df_merged)}")
    
    print(f"\nRMSE Statistics:")
    print(f"Yesterday - Mean: {df_merged['rmse_yesterday'].mean():.6f}, Std: {df_merged['rmse_yesterday'].std():.6f}")
    print(f"Competition - Mean: {df_merged['rmse_competition'].mean():.6f}, Std: {df_merged['rmse_competition'].std():.6f}")
    
    improvements = df_merged[df_merged['improvement']]
    declines = df_merged[~df_merged['improvement']]
    rank_improvements = df_merged[df_merged['rank_change'] > 0]
    rank_declines = df_merged[df_merged['rank_change'] < 0]
    
    print(f"\nPerformance Changes:")
    print(f"Teams improved RMSE: {len(improvements)} ({len(improvements)/len(df_merged)*100:.1f}%)")
    print(f"Teams declined RMSE: {len(declines)} ({len(declines)/len(df_merged)*100:.1f}%)")
    print(f"Teams improved rank: {len(rank_improvements)} ({len(rank_improvements)/len(df_merged)*100:.1f}%)")
    print(f"Teams declined rank: {len(rank_declines)} ({len(rank_declines)/len(df_merged)*100:.1f}%)")
    
    if len(improvements) > 0:
        print(f"\nBest RMSE improvements:")
        best_improvements = improvements.nsmallest(3, 'rmse_diff')
        for _, team in best_improvements.iterrows():
            print(f"  {team['team']}: {team['rmse_diff']:+.6f} (Rank: {team['rank_yesterday']}→{team['rank_competition']})")
    
    if len(rank_improvements) > 0:
        print(f"\nBest rank improvements:")
        best_rank_improvements = rank_improvements.nlargest(3, 'rank_change')
        for _, team in best_rank_improvements.iterrows():
            print(f"  {team['team']}: Rank +{team['rank_change']} ({team['rank_yesterday']}→{team['rank_competition']})")
    
    print(f"\nTop 5 Teams (Competition Period):")
    top_teams = df_merged.nsmallest(5, 'rmse_competition')
    for i, (_, team) in enumerate(top_teams.iterrows(), 1):
        rank_change = team['rank_change']
        change_symbol = "↑" if rank_change > 0 else "↓" if rank_change < 0 else "→"
        print(f"  {i}. {team['team']}: {team['rmse_competition']:.6f} "
              f"(RMSE Δ: {team['rmse_diff']:+.6f}, Rank: {team['rank_yesterday']}{change_symbol}{team['rank_competition']})")

def create_sample_files():
    """
    Create sample CSV files for testing
    """
    teams = ['Alpha', 'Bravo', 'Charlie', 'Delta', 'Echo', 
             'Foxtrot', 'Golf', 'Hotel', 'India', 'Juliet']
    
    # Yesterday data
    np.random.seed(42)
    yesterday_data = {
        'id': range(1, len(teams) + 1),
        'team': teams,
        'rmse': np.round(np.random.uniform(0.08, 0.25, len(teams)), 6)
    }
    
    # Competition data (some teams improved, some declined)
    competition_data = {
        'id': range(1, len(teams) + 1),
        'team': teams,
        'rmse': np.round(yesterday_data['rmse'] + np.random.uniform(-0.05, 0.03, len(teams)), 6)
    }
    
    df_yesterday = pd.DataFrame(yesterday_data)
    df_competition = pd.DataFrame(competition_data)
    
    df_yesterday.to_csv('subset_yesterday.csv', index=False)
    df_competition.to_csv('competition_period.csv', index=False)
    
    print("Sample files created:")
    print("  subset_yesterday.csv")
    print("  competition_period.csv")
    
    return df_yesterday, df_competition

def main():
    parser = argparse.ArgumentParser(description='Generate RMSE comparison plot with rankings')
    parser.add_argument('yesterday_file', nargs='?', help='Path to subset_yesterday.csv file')
    parser.add_argument('competition_file', nargs='?', help='Path to competition_period.csv file')
    parser.add_argument('-o', '--output', default='rmse_comparison.png', 
                       help='Output filename (default: rmse_comparison.png)')
    parser.add_argument('--create-sample', action='store_true',
                       help='Create sample CSV files for testing')
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_files()
        print("\nYou can now run: python script.py subset_yesterday.csv competition_period.csv")
    
    elif args.yesterday_file and args.competition_file:
        df_yesterday, df_competition = load_and_validate_data(args.yesterday_file, args.competition_file)
        
        if df_yesterday is not None and df_competition is not None:
            df_merged = generate_comparison_plot(df_yesterday, df_competition, args.output)
            print_statistics(df_merged)
    
    else:
        print("Please provide both CSV files or use --create-sample to generate sample files.")
        print("\nUsage examples:")
        print("  python script.py subset_yesterday.csv competition_period.csv")
        print("  python script.py subset_yesterday.csv competition_period.csv -o my_comparison.png")
        print("  python script.py --create-sample")

if __name__ == "__main__":
    main()
