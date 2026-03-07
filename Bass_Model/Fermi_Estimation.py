# Estimating the Bass model market potential (M) for the
# Nvidia DGX Spark using Fermi logic.
#
# Instead of deriving M from historical data (which is
# unreliable for a brand new product), we build up M from
# known statistics about the global developer population.
#
# Output: M in thousands of units
# References:
#   - Evans Data Corporation (2023). Global Developer Population Study.
#   - Stack Overflow Developer Survey (2023). https://survey.stackoverflow.co/2023/

global_developers = 28_700_000 # (Evans Data Corporation, 2023)

ml_share = 0.20 # (Stack Overflow Developer Survey, 2023)

serious_training_share = 0.10

ml_developers = global_developers * ml_share        # ~5.7 million ML developers
target_users = ml_developers * serious_training_share  # ~570,000 serious trainers

M = round(target_users / 1000)   # convert to thousands

print(f"Global developers:       {global_developers:,}")
print(f"ML developers (~20%):    {int(ml_developers):,}")
print(f"Serious trainers (~10%): {int(target_users):,}")
print(f"Estimated M:             {M:,} thousand units")
print(f"\nM = {M * 1000:,} units globally (conservative estimate)")
print("If adoption expands to universities and startups, M could reach 1-2 million units.")