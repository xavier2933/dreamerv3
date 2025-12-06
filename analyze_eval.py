import csv
import argparse
import os
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description='Analyze evaluation results.')
    parser.add_argument('csv_file', nargs='?', default='eval_results.csv', help='Path to the CSV file')
    args = parser.parse_args()

    if not os.path.exists(args.csv_file):
        print(f"Error: File '{args.csv_file}' not found.")
        return

    stats = {}

    print(f"Analyzing {args.csv_file}...")
    
    try:
        with open(args.csv_file, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                agent = row['Agent']
                duration = int(float(row['Duration'])) # Handle potential float strings
                reward = float(row['Reward'])
                
                if agent not in stats:
                    stats[agent] = {
                        'total': 0, 
                        'count_1': 0, 
                        'count_201': 0, 
                        'count_between': 0,
                        'rewards_gt_1': [],
                        'count_pos_gt_1': 0,
                        'all_rewards': [],
                        'all_durations': []
                    }
                
                stats[agent]['total'] += 1
                stats[agent]['all_rewards'].append(reward)
                stats[agent]['all_durations'].append(duration)
                
                if duration == 1:
                    stats[agent]['count_1'] += 1
                elif duration == 201:
                    stats[agent]['count_201'] += 1
                else:
                    stats[agent]['count_between'] += 1
                
                if duration > 1:
                    stats[agent]['rewards_gt_1'].append(reward)
                    if reward > 0:
                        stats[agent]['count_pos_gt_1'] += 1

        print("\nAnalysis Results:")
        print("-" * 130)
        print(f"{'Agent':<15} | {'Total':<8} | {'Steps=1':<10} | {'Steps=201':<10} | {'Between':<10} | {'% Between':<10} | {'Avg Reward (>1)':<16} | {'% Pos Reward (>1)':<18}")
        print("-" * 130)
        
        for agent, data in stats.items():
            total = data['total']
            c1 = data['count_1']
            c201 = data['count_201']
            c_btwn = data['count_between']
            percent = (c_btwn / total * 100) if total > 0 else 0.0
            
            rewards_gt_1 = data['rewards_gt_1']
            avg_reward_gt_1 = sum(rewards_gt_1) / len(rewards_gt_1) if rewards_gt_1 else 0.0
            
            count_pos = data['count_pos_gt_1']
            count_gt_1 = len(rewards_gt_1)
            percent_pos = (count_pos / count_gt_1 * 100) if count_gt_1 > 0 else 0.0
            
            print(f"{agent:<15} | {total:<8} | {c1:<10} | {c201:<10} | {c_btwn:<10} | {percent:<10.2f}% | {avg_reward_gt_1:<16.2f} | {percent_pos:<18.2f}%")
            
        print("-" * 130)
        print("Note: 'Between' runs are those that terminated in >1 and <201 steps.")
        print("Note: 'Avg Reward (>1)' and '% Pos Reward (>1)' exclude runs that terminated immediately.")

        # Plotting
        print("\nGenerating analysis plots (excluding 1-step episodes)...")
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        
        # Reward Plot
        ax = axes[0]
        for agent, data in stats.items():
            filtered_rewards = []
            for r, d in zip(data['all_rewards'], data['all_durations']):
                if d > 1:
                    filtered_rewards.append(r)
            ax.plot(filtered_rewards, label=agent, marker='o')
        
        ax.set_title('Episode Rewards (excluding 1-step runs)')
        ax.set_xlabel('Valid Episode Index')
        ax.set_ylabel('Total Reward')
        ax.legend()
        ax.grid(True)
        
        # Duration Plot
        ax = axes[1]
        for agent, data in stats.items():
            filtered_durations = []
            for d in data['all_durations']:
                if d > 1:
                    filtered_durations.append(d)
            ax.plot(filtered_durations, label=agent, marker='x')
            
        ax.set_title('Episode Durations (excluding 1-step runs)')
        ax.set_xlabel('Valid Episode Index')
        ax.set_ylabel('Steps')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig('eval_analysis.png')
        print("Saved plots to eval_analysis.png")

    except Exception as e:
        print(f"Error reading or parsing CSV: {e}")

if __name__ == '__main__':
    main()
