import os
import shutil
import tempfile
import pandas as pd
import torch
import io
import sys
from SLAC25.utils import split_train_val, EarlyStopping, visualize_performance

def test_split_train_val():
    #check if the function creates a validation file with the same number of rows as the test file
    # create a temporary folder to store test CSV files during testing, will be deleted after
    with tempfile.TemporaryDirectory() as tmpdir:
        train_df = pd.DataFrame({
            'id': range(10),
            'label': [0]*10
        })

        test_df = pd.DataFrame({
            'id': range(3),
            'label': [1]*3
        })

        train_path = os.path.join(tmpdir, 'train.csv')
        test_path = os.path.join(tmpdir, 'test.csv')
        val_path = os.path.join(tmpdir, 'val.csv')

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index= False)

        #call the actual function
        split_train_val(train_path, test_path, val_path)
        val_df = pd.read_csv(val_path)
        
        assert len(val_df) == 3
        assert set(val_df.columns) == set(['id','label'])

def test_early_stopping_triggers():
    #test when to stop training based on increasing validation loss
    model = torch.nn.Linear(1,1) #1 input 1 output
    early_stopper = EarlyStopping(patience=2, verbose=False)

    #first call with a validation loss of 0.5
    early_stopper(0.5, model)
    assert early_stopper.early_stop is False

    #second call with a validation loss of 0.6, patience counter = 1 now
    early_stopper(0.6, model)

    #0.7, patience counter = 2 --> should trigger early stopping
    early_stopper(0.7, model)

    assert early_stopper.early_stop is True



def test_visualize_performance_creates_plot():
    with tempfile.TemporaryDirectory() as tmpdir:
        #a fake training log
        log1 = {
            'epoch': [1,2,3],
            'train_loss': [1, 0.8, 0.6],
            'val_loss': [0.9, 0.7, 0.5],
            'train_acc': [0.5, 0.6, 0.7],    # Simulated training accuracy over epochs
            'val_acc': [0.4, 0.5, 0.6]
        }
        
        #test if the file exists
        visualize_performance(log1, tmpdir, "test_plot1.png")
        plot_path1 = os.path.join(tmpdir, "test_plot1.png")
        assert os.path.exists(plot_path1)
        assert os.path.getsize(plot_path1) > 0

        log2 = {
            'epoch': [1,2,3],
            'train_loss': [1, 0.8, 0.6],
            'val_loss': [0,0,0],
            'train_acc': [0.5, 0.6, 0.7],    # Simulated training accuracy over epochs
            'val_acc': [0,0,0]
        }
        captured_output = io.StringIO()
        original_stdout = sys.stdout
        sys.stdout = captured_output

        try:
            visualize_performance(log2, tmpdir, "test_plot2.png")
        finally:
            sys.stdout = original_stdout

        output = captured_output.getvalue()
        assert "Validation loss data is all zeros." in output
        assert "Validation accuracy data is all zeros." in output



