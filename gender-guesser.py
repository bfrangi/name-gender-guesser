import random

import nltk


def parse_arguments():
    import argparse
    import sys

    # Create parser
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', nargs='?', default=None)
    parser.add_argument('-s', '--spanish', action='store_true', help="use this flag to use only names from spain for training")

    # Parse args
    args = sys.argv[1::]
    input_file = parser.parse_args(args).input_file
    s = parser.parse_args(args).spanish
    if not input_file:
        parser.print_help()
        exit()
    return input_file, s


def gender_features(word):
    word = word.lower()
    suff_1 = - min(len(word), 1)
    suff_2 = - min(len(word), 2)
    suff_4 = - min(len(word), 4)
    suff_5 = - min(len(word), 5)
    return {
        'suffix_1': word[suff_1:],
        'suffix_2': word[suff_2:],
        'suffix_4': word[suff_4:],
        'name': word,
    }

if __name__=="__main__":
    input_file, s = parse_arguments()

    training_suffix = "_spain" if s else ""

    # Load the training and test data
    with open(f'names/male_names{training_suffix}.txt', 'r') as f:
        male = [n.strip().title() for n in f.readlines()]

    with open(f'names/female_names{training_suffix}.txt', 'r') as f:
        female = [n.strip().title() for n in f.readlines()]

    # Prepare a list of examples and corresponding class labels
    labeled_names = ([(name, 'male') for name in male] +
                    [(name, 'female') for name in female])

    # Shuffle the names to avoid any bias due to the order of the names
    random.shuffle(labeled_names)

    # Use the feature extractor to process the names data.
    featuresets = [(gender_features(n), gender) for (n, gender) in labeled_names]

    # Divide the resulting list of feature sets into a training set and a test set
    n_name = len(featuresets)
    n_train = int(n_name * 0.85)
    n_test = n_name - n_train
    train_set, test_set = featuresets[0:n_train], featuresets[n_train:n_name]

    # The training set is used to train a new "naive Bayes" classifier
    classifier = nltk.NaiveBayesClassifier.train(train_set)


    # The test set is used to evaluate the accuracy of the classifier
    errors = []
    for name, tag in test_set:
        guess = classifier.classify(gender_features(name['name']))
        if guess != tag:
            errors.append((tag, guess, name['name']))
            # print('correct={:<8} guess={:<8s} name={:<30}'.format(tag, guess, name['name']))

    # Print the accuracy of the classifier
    print("Accuracy:", nltk.classify.accuracy(classifier, test_set))

    # Load the names to infer
    with open(input_file, 'r') as f:
        names = [n.strip().title() for n in f.readlines()]

    # Record the compound names for later reconstruction, as we will only infer with the first name
    compound_names = {}
    for name in names:
        if " " in name:
            name_parts = name.split()
            first_names, other_names = name_parts[0], "".join(name_parts[1:])

            if first_names not in compound_names:
                compound_names[first_names] = [other_names]
            else:
                compound_names[first_names] += [other_names]

    # Record a list of the original names for later reconstruction
    original_names = list(set(names))
    original_names.sort()

    # Remove the compound names from the list of names to infer and sort them
    names = list(set([name.split()[0] for name in names]))
    names.sort()

    # Run the classifier and record the results
    results_male = []
    results_female = []
    for name in names:
        result = classifier.classify(gender_features(name))
        if result == "female":
            results_female.append(name)
        else:
            results_male.append(name)

    # Reconstruct the compound names
    for name in results_female:
        if name in compound_names:
            for c in compound_names[name]:
                results_female.append(name + ' ' + c)

    for name in results_male:
        if name in compound_names:
            for c in compound_names[name]:
                results_male.append(name + ' ' + c)

    results_male = [name for name in results_male if name in original_names]
    results_female = [name for name in results_female if name in original_names]

    # Convert the results to uppercase and sort them
    results_male = [name.upper() for name in results_male]
    results_female = [name.upper() for name in results_female]
    results_male.sort()
    results_female.sort()

    # Write the results to a file
    with open('results.csv', 'w') as f:
        for item in results_male:
            f.write(f"{item}, 0\n")
        for item in results_female:
            f.write(f"{item}, 1\n")
