# Prepare metric names
METRICS = ['root_mean_squared_error', 'mean_squared_error', 'r2', 'pearsonr', 'spearman']


svm_compare = {
               "title": "",
               "svm": [0.0, 0.0, 0.0, 0.0, 0.0],
               "rf": [0.0, 0.0, 0.0, 0.0, 0.0],
               "knn": [0.0, 0.0, 0.0, 0.0, 0.0]
              }
rf_compare = {
               "title": "",
               "svm": [0.0, 0.0, 0.0, 0.0, 0.0],
               "rf": [0.0, 0.0, 0.0, 0.0, 0.0],
               "knn": [0.0, 0.0, 0.0, 0.0, 0.0]
             }
knn_compare = {
               "title": "",
               "svm": [0.0, 0.0, 0.0, 0.0, 0.0],
               "rf": [0.0, 0.0, 0.0, 0.0, 0.0],
               "knn": [0.0, 0.0, 0.0, 0.0, 0.0]
              }

# Retrieve information from ''IC50_baseline_encodings.txt' file
results_baseline = open("IC50_baseline_encodings.txt", "r").readlines()

for index, line in enumerate(results_baseline):

    line = line.strip()

    if line != "":
        # If line is an entry title:
        if line[0] == ">":
            title = line.split(" ")[1]
        else:
            vals = line.split("\t")

            if vals[0] == "SVM":

                svm_mean = [float(val) for val in vals[1:]]

                rf_vals = results_baseline[index + 1].strip().split("\t")
                rf_mean = [float(val) for val in rf_vals[1:]]

                knn_vals = results_baseline[index + 2].strip().split("\t")
                knn_mean = [float(val) for val in knn_vals[1:]]

                # Compare 'spearman' values, and replace former values if new 'spearman' value is superior
                if svm_mean[4] > svm_compare['svm'][4]:
                    svm_compare["title"] = title

                    svm_compare["svm"] = svm_mean
                    svm_compare["rf"] = rf_mean
                    svm_compare["knn"] = knn_mean

                if rf_mean[4] > rf_compare['rf'][4]:
                    rf_compare["title"] = title

                    rf_compare["svm"] = svm_mean
                    rf_compare["rf"] = rf_mean
                    rf_compare["knn"] = knn_mean

                if knn_mean[4] > knn_compare['knn'][4]:
                    knn_compare["title"] = title

                    knn_compare["svm"] = svm_mean
                    knn_compare["rf"] = rf_mean
                    knn_compare["knn"] = knn_mean

# Print best values for each baseline model
print(f"BEST MODELS\n-----------------\n")

print(f"Best SVM model - {svm_compare['title']}:\n---")
print(f"SVM:\t{svm_compare['svm']}")
print(f"RF: \t{svm_compare['rf']}")
print(f"KNN: \t{svm_compare['knn']}")

print()

print(f"Best RF model - {rf_compare['title']}:\n---")
print(f"SVM:\t{rf_compare['svm']}")
print(f"RF: \t{rf_compare['rf']}")
print(f"KNN: \t{rf_compare['knn']}")

print()

print(f"Best KNN model - {knn_compare['title']}:\n---")
print(f"SVM:\t{knn_compare['svm']}")
print(f"RF: \t{knn_compare['rf']}")
print(f"KNN: \t{knn_compare['knn']}")
