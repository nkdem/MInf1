from pathlib import Path
from tabulate import tabulate

def compare_samples():
    # Define paths
    our_path = Path("/Users/nkdem/Downloads/HEAR-DS/Down-Sampled")
    
    # Paper's numbers from the table
    paper_counts = {
        'CocktailParty': 667,
        'InterfereringSpeakers': 1481,
        'InTraffic': 1000,
        'InVehicle': 584 + 511,
        'Music': 1496 + 1495,
        'QuietIndoors': 525 + 426,
        'ReverberantEnvironment': 315 + 692,
        'WindTurbulence': 595 + 439 
    }
    
    # Get our counts
    our_counts = {}
    for env_name in paper_counts.keys():
        env_path = our_path / env_name / 'Background'
        if env_path.exists():
            # Count wav files and divide by 2 (since we have L and R channels)
            count = len(list(env_path.glob('*.wav'))) // 2
            our_counts[env_name] = count
        else:
            our_counts[env_name] = 0
    
    # Prepare table data
    table_data = []
    for env_name in paper_counts.keys():
        paper_count = paper_counts[env_name]
        our_count = our_counts[env_name]
        difference = our_count - paper_count
        percentage = (our_count / paper_count * 100) if paper_count > 0 else 0
        
        table_data.append([
            env_name,
            paper_count,
            our_count,
            difference,
            f"{percentage:.1f}%"
        ])
    
    # Print table
    headers = ["Environment", "Paper Count", "Our Count", "Difference", "Percentage"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Print summary
    paper_total = sum(paper_counts.values())
    our_total = sum(our_counts.values())
    print(f"\nTotal in paper: {paper_total}")
    print(f"Total in our dataset: {our_total}")
    print(f"Overall difference: {our_total - paper_total}")
    print(f"Overall percentage: {(our_total/paper_total*100):.1f}%")

if __name__ == "__main__":
    compare_samples()
