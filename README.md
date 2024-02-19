# Name Gender Guesser

The Name Gender Guesser is a simple Python script designed to predict the gender associated with a given name.

## Installation

To install the required dependencies, execute the following command in your terminal:

```bash
pip install -r requirements.txt
```

## Usage

To determine the likely gender associated with a list of names, follow this usage example:

```bash
python3 gender-guesser.py 'names_to_infer.txt'
```

Ensure that the file `names_to_infer.txt` contains a list of names, with each name on a separate line. The program is not case-sensitive.

### Optional Parameters

You can utilize optional parameters to refine the prediction process. For instance, the `-s` or `--spanish` flag enables training the classification algorithm specifically with names most commonly found in the Spanish population.

```bash
python3 gender-guesser.py 'names_to_infer.txt' -s
```

This optional flag tailors the prediction to better suit Spanish naming conventions.
