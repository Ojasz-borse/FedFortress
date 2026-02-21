from src.main import run_federated_training

for result in run_federated_training(
        aggregation="FedAvg",
        num_clients=3,
        malicious_clients=1,
        rounds=2):

    print("Round:", result["round"])
    print("Accuracy:", result["accuracy"])
    print("-" * 40)