<div style="text-align:center">
	<img src="figures/ji_logo.png" alt="Jilogo" style="zoom:60%;" />
</div>
<center>
	<h2>
		VE406 Applied Regression Analysis using R
	</h2>
</center> 
<center>
	<h3>
		Tesla Stock Prediction
	</h3>
</center>
<center>
   <h4>
       FA 2020
    </h4> 
</center>

------------------------------------------

### Abstract

This project is used to help our VE406 Project team work together. 

Once we open source the code for `20FA VE406 Project` and if you want to refer to our work, please follow the Joint Institute’s honor code and don’t plagiarize these codes directly.

### Dependency Requirements

1. All platforms supported, `windows`, `MacOC`, `linux`. 
2. `python>=3.7` should be installed. 
3. To install the dependent `python` packages, please run the following commands:
   
   ```bash
   conda create -n tesla python==3.7
   pip install --upgrade pip
   pip install -r requirements.txt
   conda activate tesla
   ```

### Project Usage

1. `ARIMA-GARCH.py`: main project file of final model, Seasonal-ARIMA-GARCH.
2. `pure-ARIMA.py`: main project file of Benchmark model, Seasonal-ARIMA.
   
For **TA** grading, please run the following command:

```
python3 ARIMA-GARCH.py --period 7 --split_ratio 1
```
After tuning, above choice of some hyperparameters give the optimized results.

For more usage, please run the commmand:

```bash
python3 XXX.py --help
```

The detailed instruction will be provided in std output.

### Code Style

**Example**:

```python
def sch_random(round_idx, time_counter):
    # set the seed
    np.random.seed(round_idx)

    # random sample clients
    cars = list(channel_data[channel_data['Time'] == time_counter]['Car'])
    client_indexes = []
    if cars:
        client_indexes = list(np.random.choice(cars, max(int(len(cars) / 2), 1), replace=False).ravel())
    logger.info("client_indexes = " + str(client_indexes) + "; time_counter = " + str(time_counter))

    # random local iterations
    local_itr = np.random.randint(2) + 1

    s_client_indexes = str(list(client_indexes))[1:-1].replace(',', '')
    s_local_itr = str(local_itr)

    return s_client_indexes + "," + s_local_itr
```

1. Plz avoid meaningless combination of letters like `a` or `abc` when naming variables. Name of variable should be meaningful. 
3. Plz add appropriate indentation and blank lines to your code.
4. Plz add enough comments to help others understand your code.
4. If there is any hyper parameter in your code or global parameter, plz use `argparse` or create another file called `config.py` to store them. For example, [config.py](https://github.com/zzp1012/federated-learning-environment/blob/master/fedavg/config.py) and [argparse](https://github.com/zzp1012/federated-learning-environment/blob/master/fedavg/scheduler.py)

### Git Usage

Here are some simple instructions about how to use `Git`.

1. If you want to download the whole project, run following command.

```bash
git clone https://github.com/zzp1012/VE370-Project2.git
```

2. If you want add files to our local git project and remote git project on `github`, run following command.

```bash
# Firstly, plz avoid adding files to master branch on github directly. You can create your own branch locally and remotely.

git branch zzp1012 # create my local branch. Here I name the branch as 'zzp1012'. If you have already created a branch, you can jump to next command.

git checkout zzp1012 # switch to 'zzp1012' branch.

git add * # add all the files to local branch 'zzp1012'.

git commit -m "update" # confirm to add files to local branch 'zzp1012'

git push origin zzp1012 # create branch 'zzp1012' remotely on github and copy your the content on your local branch 'zzp1012' to the remote 'zzp1012'.
```

3. If you want to synchronize files on remote project on `github`, you should run:

```bash
git pull origin master # synchronize files on remote master branch.
git pull origin "you branch name" # the 'master' can be replaced by the name of the other branch created on remote project on github, then you can synchronize files on the specific remote branch.
```

### Reference

[1] Zhu, T., 2020. *Ve406 Applied Regression Analysis using R Project*.

---------------------------------------------------------------

<center>
    UM-SJTU Joint Institute 交大密西根学院
</center>