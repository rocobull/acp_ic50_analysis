# Prepare metric names
METRICS = ['root_mean_squared_error', 'mean_squared_error', 'r2', 'pearsonr', 'spearman']


# Retrieve information from ''IC50_baseline_encodings.txt' file
results_baseline = open("IC50_baseline_encodings.txt", "r").readlines()

print(f"\n\nBEST MODELS\n-----------------\n")

for desc in ["nlf", "esm1", "esm2_8M", "esm2_35M", "esm2_150M", "esm2_650M", "esm2_3B", "protbert", "z-scales"]:

    svm_compare = {
                   "title": "",
                   "svm": [0.0, 0.0, 0.0, 0.0, 0.0]
                  }
    rf_compare = {
                   "title": "",
                   "rf": [0.0, 0.0, 0.0, 0.0, 0.0]
                 }
    knn_compare = {
                   "title": "",
                   "knn": [0.0, 0.0, 0.0, 0.0, 0.0]
                  }

    for index, line in enumerate(results_baseline):

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

                    svm_scores = [float(val) for val in vals[1:]]

                    rf_vals = results_baseline[index + 1].strip().split("\t")
                    rf_scores = [float(val) for val in rf_vals[1:]]

                    knn_vals = results_baseline[index + 2].strip().split("\t")
                    knn_scores = [float(val) for val in knn_vals[1:]]

                    # Compare 'spearman' values, and replace former values if new 'spearman' value is superior
                    if svm_scores[4] > svm_compare['svm'][4]:
                        svm_compare["title"] = title
                        svm_compare["svm"] = svm_scores

                    if rf_scores[4] > rf_compare['rf'][4]:
                        rf_compare["title"] = title
                        rf_compare["rf"] = rf_scores

                    if knn_scores[4] > knn_compare['knn'][4]:
                        knn_compare["title"] = title
                        knn_compare["knn"] = knn_scores

    # Print best values for each baseline model, in regard to each encoding preset
    print(f"\n'{desc}'\n{'-' * len(desc)}--\n")

    print(f"Best SVM model - {svm_compare['title']}:\n---")
    print(f"SVM:\t{svm_compare['svm']}")

    print()

    print(f"Best RF model - {rf_compare['title']}:\n---")
    print(f"RF: \t{rf_compare['rf']}")

    print()

    print(f"Best KNN model - {knn_compare['title']}:\n---")
    print(f"KNN: \t{knn_compare['knn']}")

    print()
