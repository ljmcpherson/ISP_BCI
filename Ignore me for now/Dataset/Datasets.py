import os
import requests
import argparse

def download_miller(root):
    url = 'https://osf.io/ksqv8/download'
    fname = os.path.join(root, 'motor_imagery.npz')

    if not os.path.isfile(fname):
        try:
            r = requests.get(url)
            r.raise_for_status()
        except requests.ConnectionError:
            print('!!! Failed to download data !!!')
        except requests.HTTPError:
            print('!!! Failed to download data with status code:', r.status_code)
        else:
            with open(fname, 'wb') as fid:
                fid.write(r.content)
            print(f'Downloaded {fname} successfully.')
    else:
        print(f'File already exists')

def main(dataset, root):

    dataset_list = ['miller']

    if dataset.lower() == 'miller':
        download_miller(root)

    elif dataset.lower() == 'none':
        download_miller(root) 
        # Will add more later
    else:
        print(f'Error: Dataset was not recognized. Please choose a dataset among [{dataset_list}]')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download datasets.')
    parser.add_argument('--dataset', type=str, required=False, help='Name of the dataset.', default='miller')
    parser.add_argument('--root', type=str, required=False, help='Root path to save files.', default='./')
    
    args = parser.parse_args()
    
    main(args.dataset, args.root)