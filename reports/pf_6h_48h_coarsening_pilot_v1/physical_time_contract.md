# Physical-time contract

The converter-derived T380 pilot parameters are recorded in the cluster
unit_contract.json and time_contract.txt. The first pilot segment reports:

temperature_C=380
dt_code=0.02
t_real_unit_s=49.54630476715921
dt_physical_s=0.9909260953431841
steps_per_physical_hour=3633

The effective fixture is the experimental 6 h handoff. The serialized chain
advances 42 physical hours to 48 h, with a checkpoint after every physical
hour.
