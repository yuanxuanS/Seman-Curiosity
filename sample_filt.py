
import pickle

uncertain_pth = "/home/wpp/Seman-Curiosity/data/vsqf_test_val5/uncertainty_sum_cam.pkl"
with open(uncertain_pth, "rb") as f:
    uncertain = pickle.load(f)
    
for uncertain_func in [0, 1, 2]:
    for samples in [3000]:
        if uncertain_func == 1:
            sorted_uncertain = sorted(uncertain, key=lambda x: x[uncertain_func + 3].sum())
        else:
            sorted_uncertain = sorted(uncertain, key=lambda x: x[uncertain_func + 3])
            
        print("==================ucnertain_func", uncertain_func, "samples", samples)
        for i in range(samples):
            if i == 500:
                print("-----500-----")
            if i == 1000:
                print("-----1000-----")
            if i == 1500:
                print("-----1500-----")
            if i == 2000:
                print("-----2000-----")
            env, ep, step = sorted_uncertain[-i][:3]
            print(f"env {env} epi {ep} step {step}")