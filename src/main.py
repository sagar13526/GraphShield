"""Run the complete synthetic fraud-detection workflow."""
from generate_data import generate
from train import main as train
from visualize import main as visualize

def main():
    print("\n[1/3] Generating synthetic transaction graph...")
    generate()
    print("\n[2/3] Training models...")
    train()
    print("\n[3/3] Creating visualizations...")
    visualize()
    print("\nAll done. Check the results/ folder and README.md.")

if __name__ == "__main__":
    main()
