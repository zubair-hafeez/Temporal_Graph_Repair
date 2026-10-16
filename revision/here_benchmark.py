"""Author-code HERE Transformer on its closed-case missing-ID task.

This integration preserves the supplied model, dataset tensorization, and greedy
decoder. The headless runner and explicitly separated masked-event evaluation
are ours. No unavailable expert rules are invented. This is a named supported
Transformer configuration, not reproduction of every published HERE setting.
"""
import ast
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

from tgrepair.data import Graph, load_graph, save_graph, sha256
from tgrepair.metrics import traceability
from revision.vendor.here_dataset import InOutDataset, causal_mask
from revision.vendor.here_model import build_transformer, Transformer

ROOT = Path(__file__).resolve().parents[1]
SPECIAL = ['[UNK]', '[PAD]', '[SOS]', '[EOS]', '[NONE]']


def author_greedy_decoder():
    """Execute exactly one licensed author function without importing the GUI."""
    source = ROOT / 'revision/vendor/here_train.py'
    tree = ast.parse(source.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'greedy_decode')
    module = ast.Module(body=[function], type_ignores=[])
    scope = {'torch': torch, 'Transformer': Transformer, 'Tokenizer': Tokenizer, 'causal_mask': causal_mask}
    exec(compile(module, str(source), 'exec'), scope)
    return scope['greedy_decode']


def project_single_case(graph, root_type, max_roots):
    roots = sorted((o for o, t in graph.objects.items() if t == root_type),
                   key=lambda o: hashlib.sha256(('here_development_v1:' + o).encode()).hexdigest())[:max_roots]
    selected = set(roots)
    events = []
    objects = {}
    for e in graph.events:
        case = [o for o in e.participants if o in selected]
        if not case:
            continue
        if len(case) != 1:
            raise ValueError('Native task requires exactly one case per event')
        participants = tuple(o for o in e.participants if o in selected or graph.objects[o] == 'Resource')
        objects.update({o: graph.objects[o] for o in participants})
        # Match the activity/resource/time inputs used in the native comparison;
        # object-child IDs, vendor context, and richer event fields are absent.
        events.append(replace(e, participants=participants, attributes={}))
    result = Graph(graph.name + '_single_case', objects, events)
    result.validate()
    return result


def prepare_records(graph, root_type, missing_rate, seed):
    rng = random.Random(124000 + seed)
    cases = sorted(o for o, t in graph.objects.items() if t == root_type)
    case_number = {o: str(i + 1) for i, o in enumerate(cases)}
    times = [e.time for e in graph.events]
    t0, span = min(times), max(1., max(times) - min(times))
    rows, hidden, truth = [], set(), {}
    observed_events = []
    for e in graph.events:
        case = next(o for o in e.participants if graph.objects[o] == root_type)
        resources = sorted(o for o in e.participants if graph.objects[o] == 'Resource')
        # Stable categorical tokens contain no semantic target information.
        activity = 'activity_' + hashlib.sha256(e.kind.encode()).hexdigest()[:16]
        resource = 'resource_' + hashlib.sha256('|'.join(resources).encode()).hexdigest()[:16]
        missing = rng.random() < missing_rate
        if missing:
            hidden.add(e.uid)
        truth[e.uid] = case
        rows.append({'uid': e.uid, 'activity': activity, 'resource': resource,
                     'time': (e.time - t0) / span, 'label': '[NONE]' if missing else case_number[case]})
        observed_events.append(replace(e, participants=tuple(o for o in e.participants if not (missing and o == case))))
    observed = Graph(graph.name + '_missing', dict(graph.objects), observed_events)
    reference = {'identity': {o: o for o in graph.objects},
                 'event_participation': {e.uid: list(e.participants) for e in graph.events}}
    return rows, hidden, truth, {v: k for k, v in case_number.items()}, observed, reference


def sequence(rows):
    return {'discrete': ' '.join([r['activity'] for r in rows] + [r['resource'] for r in rows]),
            'continuous': ' '.join(str(r['time']).replace('-', '_') for r in rows),
            'target': ' '.join(r['label'] for r in rows)}


def tokenizer(sequences, key):
    result = Tokenizer(models.WordLevel(unk_token='[UNK]'))
    result.pre_tokenizer = pre_tokenizers.Whitespace()
    result.train_from_iterator((s[key] for s in sequences),
                               trainer=trainers.WordLevelTrainer(special_tokens=SPECIAL))
    return result


def make_dataset(sequences, source_tokenizer, target_tokenizer):
    return InOutDataset(sequences, source_tokenizer, target_tokenizer,
                         'discrete', 'continuous', 'target', 17, 2, 1)


def train_development(domain, seed=0, missing_rate=.1, epochs=2, max_roots=128, device_name='cuda:0'):
    """Pilot only; no sealed native-comparison partition can be opened here."""
    torch.set_num_threads(4)
    if device_name.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; use the approved paper revision runner')
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    source = ROOT / 'revision/data/cohorts_v1' / domain / 'development/reference_graph.json.gz'
    manifest = json.loads((ROOT / 'revision/reports' / f'cohorts_{domain}_v1.json').read_text())
    root_type = manifest['root_type']
    clean = project_single_case(load_graph(source), root_type, max_roots)
    rows, hidden, truth, number_case, observed, reference = prepare_records(clean, root_type, missing_rate, seed)
    # Correct remaining case IDs provide supervision. Withheld IDs never appear
    # in fitting or validation targets. Complete known rows form author-style
    # sliding windows; inference sees the original full temporal stream.
    known = [r for r in rows if r['uid'] not in hidden]
    sequences = [sequence(known[i:i + 15]) for i in range(len(known) - 14)]
    rng = random.Random(125000 + seed)
    permutation = list(range(len(sequences)))
    rng.shuffle(permutation)
    usable = permutation[:int(.9 * len(permutation))]
    split = int(.9 * len(usable))
    fitting = [sequences[i] for i in usable[:split]]
    validation = [sequences[i] for i in usable[split:]]
    if not fitting or not validation:
        raise ValueError('Insufficient known-event windows')
    source_tokenizer = tokenizer([sequences[i] for i in usable], 'discrete')
    target_tokenizer = tokenizer([sequences[i] for i in usable], 'target')
    ds = make_dataset(fitting, source_tokenizer, target_tokenizer)
    train_loader = DataLoader(ds, batch_size=8, shuffle=True, generator=torch.Generator().manual_seed(seed))
    valid_loader = DataLoader(make_dataset(validation, source_tokenizer, target_tokenizer), batch_size=8)
    config = {'d_model': 512, 'num_layers': 3, 'h': 8, 'dropout': .1, 'd_ff': 2048}
    model = build_transformer(source_tokenizer.get_vocab_size(), target_tokenizer.get_vocab_size(), 32, 17, **config)
    device = torch.device(device_name)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, eps=1e-9)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=source_tokenizer.token_to_id('[PAD]'), label_smoothing=.1)
    name = f'here_development_{domain}_seed{seed}_missing{int(100 * missing_rate)}_roots{max_roots}_epochs{epochs}_v1'
    destination = ROOT / 'revision/reports' / (name + '.json')
    if destination.exists():
        raise FileExistsError(destination)
    artifacts = ROOT / 'revision/checkpoints' / name
    artifacts.mkdir()
    source_tokenizer.save(str(artifacts / 'source_tokenizer.json'))
    target_tokenizer.save(str(artifacts / 'target_tokenizer.json'))
    save_graph(observed, artifacts / 'observed.json.gz')
    report = {'status': 'running_here_development_pilot', 'name': name, 'domain': domain,
              'seed': seed, 'missing_rate': missing_rate, 'epochs': epochs, 'config': config,
              'source_graph_sha256': sha256(source), 'runner_sha256': sha256(__file__),
              'author_commit': '814b1db618aaa9f14e6f471504699ade9fcbf203',
              'author_files': {p.name: sha256(p) for p in (ROOT / 'revision/vendor').glob('here_*')},
              'observed_events': len(rows), 'cases': len(number_case), 'hidden_test_events': len(hidden),
              'known_supervision_events': len(known), 'fit_windows': len(fitting), 'validation_windows': len(validation),
              'source_vocab': source_tokenizer.get_vocab_size(), 'target_vocab': target_tokenizer.get_vocab_size(),
              'parameter_count': sum(p.numel() for p in model.parameters()), 'training': [],
              'limits': ['Development pilot, not reserved confirmation and not a published-score reproduction.',
                         'Author model, tensor dataset, and greedy decoder preserved. Headless orchestration and evaluation split are explicit adaptations.',
                         'No supplied expert attributes or declarative rules; name this HERE-Transformer without expert rules.',
                         'Window validation overlaps known fitting events and is diagnostic only; no masked test identifier enters any training window.',
                         'This closed-case track uses known target IDs as supervision and is separate from the verified-history adaptation budget.',
                         'A two-epoch pilot checks runtime only; final method comparisons require the full declared training budget.']}
    start = time.monotonic()
    destination.write_text(json.dumps(report, indent=2) + '\n')

    def logits(batch):
        source_mask = batch['encoder_mask'].to(device)
        encoder = model.encode(batch['encoder_input'].to(device), source_mask)
        encoder = model.cont_enrichment(encoder, batch['continuous_input'].to(device))
        decoder = model.decode(encoder, batch['continuous_mask'].to(device),
                               batch['decoder_input'].to(device), batch['decoder_mask'].to(device))
        return model.project(decoder)

    for epoch in range(epochs):
        model.train()
        loss_sum = 0.
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            projected = logits(batch)
            labels = batch['label'].to(device)
            loss = loss_fn(projected.reshape(-1, target_tokenizer.get_vocab_size()), labels.reshape(-1))
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite HERE loss')
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        model.eval()
        valid_sum = 0.
        with torch.no_grad():
            for batch in valid_loader:
                projected = logits(batch)
                valid_sum += loss_fn(projected.reshape(-1, target_tokenizer.get_vocab_size()),
                                     batch['label'].to(device).reshape(-1)).item()
        row = {'epoch': epoch + 1, 'fit_loss': loss_sum / len(train_loader),
               'validation_loss': valid_sum / len(valid_loader), 'seconds': time.monotonic() - start}
        report['training'].append(row)
        destination.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({'name': name, **row}), flush=True)
    checkpoint = artifacts / 'model.pt'
    torch.save({'model': model.state_dict(), 'config': config, 'epochs': epochs}, checkpoint)
    report['model_sha256'] = sha256(checkpoint)
    greedy = author_greedy_decoder()
    predicted = {}
    probabilities = {}
    chunks = [rows[i:i + 15] for i in range(0, len(rows), 15)]
    infer = make_dataset([sequence(chunk) for chunk in chunks], source_tokenizer, target_tokenizer)
    model.eval()
    with torch.no_grad():
        for chunk, batch in zip(chunks, DataLoader(infer, batch_size=1)):
            tokens, scores, _ = greedy(model, batch['encoder_input'].to(device), batch['encoder_mask'].to(device),
                                      batch['continuous_input'].to(device), batch['continuous_mask'].to(device),
                                      target_tokenizer, len(chunk) + 2, device,
                                      {'continuous_input_attributes': ['Timestamp'], 'missing_placeholder': '[NONE]'}, 0.)
            decoded = [target_tokenizer.id_to_token(int(t)) for t in tokens.cpu().tolist()[1:]]
            for row, token, score in zip(chunk, decoded, scores):
                if row['uid'] in hidden:
                    predicted[row['uid']] = number_case.get(token)
                    probabilities[row['uid']] = score
    # Keep trusted existing IDs exactly as the author integration does.
    repaired_events = []
    for e in observed.events:
        case = predicted.get(e.uid)
        repaired_events.append(replace(e, participants=tuple(sorted(set(e.participants) | ({case} if case else set())))))
    repaired = Graph(observed.name + '_here', dict(observed.objects), repaired_events)
    correct = sum(predicted.get(uid) == truth[uid] for uid in hidden)
    assigned = sum(predicted.get(uid) is not None for uid in hidden)
    report['heldout_missing_id'] = {'correct': correct, 'assigned': assigned, 'hidden': len(hidden),
                                   'accuracy_all_missing': correct / max(1, len(hidden)),
                                   'precision_assigned': correct / assigned if assigned else None,
                                   'coverage': assigned / max(1, len(hidden)),
                                   'true_cases_unseen_in_known_records': len(set(truth[u] for u in hidden)
                                       - {number_case[r['label']] for r in known})}
    report['history_before'] = traceability(observed, reference, {root_type})
    report['history_after'] = traceability(repaired, reference, {root_type})
    report['seconds'] = time.monotonic() - start
    report['status'] = 'completed_here_development_pilot'
    (artifacts / 'predictions.json').write_text(json.dumps({'predicted': predicted, 'probabilities': probabilities}) + '\n')
    save_graph(repaired, artifacts / 'repaired.json.gz')
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'name': name, 'result': report['heldout_missing_id'], 'seconds': report['seconds']}), flush=True)

