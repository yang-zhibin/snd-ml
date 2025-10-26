import torch
import uproot
import gzip
from argparse import ArgumentParser

def process_column_data(hit_array, feature_array, hit_column_name, feature_column_name, idx):
    # Remove the prefix 'Hits/Hits.' or 'vetoHits/vetoHits.' for more readable keys
    hit_tensors = {
        col.split('.')[-1]: torch.tensor(hit_array[col][idx], dtype=torch.float32) for col in hit_column_name
    }
    
    one_event_data = {
        "hitFeature": hit_tensors,
    }

    # Only process feature array if it's provided
    if feature_array is not None:
        feature_tensors = {
            col: torch.tensor(feature_array[col][idx], dtype=torch.float32) for col in feature_column_name
        }
        one_event_data["eventFeatures"] = feature_tensors

    return one_event_data


def main(args):
    hit_column_name = [
        'Hits/Hits.detType', 'Hits/Hits.orientation', 'Hits/Hits.x1', 'Hits/Hits.y1',
        'Hits/Hits.z1', 'Hits/Hits.x2', 'Hits/Hits.y2', 'Hits/Hits.z2',
        'Hits/Hits.qdc','Hits/Hits.hitTime'
    ]
    id_column_name = [
        'Id/pdgCode', 'Id/eventId', 'Id/runId'
    ]
    feature_column_name = ['train_weight']

    tree_name = 'sndData'
    chunk_size = int(args.chunk_size)
    hit_path = args.hit_path
    feature_path = args.feature_path
    out_file = args.out_file

    with uproot.open(hit_path) as hit_file:
        if tree_name not in hit_file:
            print(f"Tree {tree_name} not found in hit file")
            return
        hit_tree = hit_file[tree_name]
        total_events = hit_tree.num_entries
        feature_tree = None
        feature_array = None

        if feature_path:
            with uproot.open(feature_path) as feature_file:
                if tree_name not in feature_file:
                    print(f"Tree {tree_name} not found in feature file")
                    return
                feature_tree = feature_file[tree_name]
                if hit_tree.num_entries != feature_tree.num_entries:
                    raise ValueError("Mismatch between hit_tree and feature_tree entry counts")

        print(f"Total events: {total_events}")
        event_data = []

        for start in range(0, total_events, chunk_size):
            end = min(start + chunk_size, total_events)
            print(f"Processing chunk: Events {start} to {end}")

            hit_array = hit_tree.arrays(hit_column_name, entry_start=start, entry_stop=end)
            id_array = hit_tree.arrays(id_column_name, entry_start=start, entry_stop=end)
            
            if feature_tree:
                feature_array = feature_tree.arrays(feature_column_name, entry_start=start, entry_stop=end)
                num_events_in_chunk = len(feature_array[feature_column_name[0]])
            else:
                num_events_in_chunk = len(hit_array[hit_column_name[0]])

            for idx in range(num_events_in_chunk):
                one_event_data = process_column_data(hit_array, feature_array, 
                                                     hit_column_name, 
                                                     feature_column_name, idx)
                
                one_event_data["pdgCode"] = id_array['Id/pdgCode'][idx]
                one_event_data["runId"] = id_array['Id/runId'][idx]
                one_event_data["eventId"] = id_array['Id/eventId'][idx]
                
                #print(one_event_data)

                event_data.append(one_event_data)


        with gzip.open(out_file, 'wb') as f:
            torch.save(event_data, f)
        print(f"Saved data to {out_file}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hit", "--hitPath", dest="hit_path", help="Hit data file path", required=True)
    parser.add_argument("-f", "--featurePath", dest="feature_path", help="Optional feature file path", required=False)
    parser.add_argument("-o", "--outfile", dest="out_file", help="Output .pt.gz file", required=True)
    parser.add_argument("-c", "--chunkSize", dest="chunk_size", help="Max chunk size", default=1e5, type=int)

    args = parser.parse_args()
    main(args)


# python root_2_pt.py -hit /eos/experiment/sndlhc/users/zhibin/MC_neutrino/volTarget_100fb-1/0/vetoFree_hit_MC_neutrino_volTarget_100fb-1_0.root -o /eos/experiment/sndlhc/users/zhibin/MC_neutrino/volTarget_100fb-1/0/vetoFree_pt_hit_MC_neutrino_volTarget_100fb-1_0.pt.gz