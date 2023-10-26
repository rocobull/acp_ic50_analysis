# Prepare metric names
METRICS = ['accuracy', 'recall', 'specificity', 'precision', 'mcc']

# Retrieve information from '740_baseline.txt' and '240_baseline.txt' files
baseline_740 = open("results/740_baseline.txt", "r").readlines()
baseline_240 = open("results/240_baseline.txt", "r").readlines()

for lines, data in zip(*[[baseline_740, baseline_240], ["740", "240"]]):

    print(f"\n\nBEST MODELS ({data})\n-----------------\n")

    for desc in ["all", "performance", "physico-chemical"]:

        svm_compare = {
                       "title": "",
                       "svm": {"mean": [0.0, 0.0, 0.0, 0.0, 0.0], "sd": [0.0, 0.0, 0.0, 0.0, 0.0]}
                       }
        rf_compare = {
                       "title": "",
                       "rf": {"mean": [0.0, 0.0, 0.0, 0.0, 0.0], "sd": [0.0, 0.0, 0.0, 0.0, 0.0]}
                       }
        nb_compare = {
                       "title": "",
                       "nb": {"mean": [0.0, 0.0, 0.0, 0.0, 0.0], "sd": [0.0, 0.0, 0.0, 0.0, 0.0]}
                       }

        for index, line in enumerate(lines):
            line = line.strip()

            if line != "":
                # If line is an entry title:
                if line[0] == ">":
                    title = line.split(" ")[1]
                else:
                    if title[2:] != desc:
                        continue
                    vals = line.split("\t")

                    if vals[0] == "SVM":

                        svm_mean = [float(val) for val in vals[1:6]]
                        # Ignore '(' and ')' characters for first and last SD value, respectively
                        svm_sd = [float(val[1:]) if ix == 0 else
                                   float(val[:-1]) if ix == 4 else
                                   float(val)
                                   for ix,val in enumerate(vals[6:])]

                        rf_vals = lines[index + 1].strip().split("\t")
                        rf_mean = [float(val) for val in rf_vals[1:6]]
                        rf_sd = [float(val[1:]) if ix == 0 else
                                 float(val[:-1]) if ix == 4 else
                                 float(val)
                                 for ix, val in enumerate(rf_vals[6:])]

                        nb_vals = lines[index + 2].strip().split("\t")
                        nb_mean = [float(val) for val in nb_vals[1:6]]
                        nb_sd = [float(val[1:]) if ix == 0 else
                                 float(val[:-1]) if ix == 4 else
                                 float(val)
                                 for ix, val in enumerate(nb_vals[6:])]

                        # Compare 'accuracy' values, and replace former values if new 'accuracy' value is superior
                        if svm_mean[0] > svm_compare['svm']["mean"][0]:
                            svm_compare["title"] = title

                            svm_compare["svm"]["mean"] = svm_mean
                            svm_compare["svm"]["sd"] = svm_sd

                        if rf_mean[0] > rf_compare['rf']["mean"][0]:
                            rf_compare["title"] = title

                            rf_compare["rf"]["mean"] = rf_mean
                            rf_compare["rf"]["sd"] = rf_sd

                        if nb_mean[0] > nb_compare['nb']["mean"][0]:
                            nb_compare["title"] = title

                            nb_compare["nb"]["mean"] = nb_mean
                            nb_compare["nb"]["sd"] = nb_sd

        # Print best values for each baseline model, in regard to each descriptor preset
        print(f"'{desc}'\n{'-'*len(desc)}--\n")

        print(f"Best SVM model - {svm_compare['title']}:\n---")
        print(f"{svm_compare['svm']['mean']} ({svm_compare['svm']['sd']})")

        print()

        print(f"Best RF model - {rf_compare['title']}:\n---")
        print(f"{rf_compare['rf']['mean']} ({rf_compare['rf']['sd']})")

        print()

        print(f"Best NB model - {nb_compare['title']}:\n---")
        print(f"{nb_compare['nb']['mean']} ({nb_compare['nb']['sd']})")

        print("\n\n")
