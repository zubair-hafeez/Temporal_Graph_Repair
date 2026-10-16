"""Frozen native-task HERE training; orchestration adapted from the audited development runner."""
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


from revision.here_benchmark import project_single_case, prepare_records, sequence, tokenizer, make_dataset, author_greedy_decoder
from revision.native_protocol import verify_protocol

def train_native(domain, seed=0, missing_rate=.3, epochs=30, max_roots=2048, device_name='cuda:0'):
    """Fixed native evaluation authorized only by the pinned protocol."""
    verify_protocol(domain, seed, missing_rate, epochs, max_roots)
    torch.set_num_threads(4)
    if device_name.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; use the approved paper revision runner')
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    source = ROOT / 'revision/data/native_cohort_v1' / domain / 'reference_graph.json.gz'
    manifest = json.loads((ROOT / 'revision/reports' / f'native_cohort_{domain}_v1.json').read_text())
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
    name = f'here_native_{domain}_seed{seed}_missing{int(100 * missing_rate)}_roots{max_roots}_epochs{epochs}_v1'
    destination = ROOT / 'revision/reports' / (name + '.json')
    if destination.exists():
        raise FileExistsError(destination)
    artifacts = ROOT / 'revision/checkpoints' / name
    artifacts.mkdir()
    source_tokenizer.save(str(artifacts / 'source_tokenizer.json'))
    target_tokenizer.save(str(artifacts / 'target_tokenizer.json'))
    save_graph(observed, artifacts / 'observed.json.gz')
    report = {'status': 'running_here_native_confirmation', 'name': name, 'domain': domain,
              'seed': seed, 'missing_rate': missing_rate, 'epochs': epochs, 'config': config,
              'source_graph_sha256': sha256(source), 'runner_sha256': sha256(__file__),
              'observed_graph_sha256': sha256(artifacts / 'observed.json.gz'),
              'author_commit': '814b1db618aaa9f14e6f471504699ade9fcbf203',
              'protocol_sha256': sha256(ROOT / 'revision/docs/NATIVE_CONFIRMATION_PROTOCOL_V1.json'),
              'author_files': {p.name: sha256(p) for p in (ROOT / 'revision/vendor').glob('here_*')},
              'observed_events': len(rows), 'cases': len(number_case), 'hidden_test_events': len(hidden),
              'known_supervision_events': len(known), 'fit_windows': len(fitting), 'validation_windows': len(validation),
              'source_vocab': source_tokenizer.get_vocab_size(), 'target_vocab': target_tokenizer.get_vocab_size(),
              'parameter_count': sum(p.numel() for p in model.parameters()), 'training': [],
              'limits': ['Reserved native-task confirmation, with an explicit adaptation of author code rather than a published-score reproduction.',
                         'Author model, tensor dataset, and greedy decoder preserved. Headless orchestration and evaluation split are explicit adaptations.',
                         'No supplied expert attributes or declarative rules; name this HERE-Transformer without expert rules.',
                         'Window validation overlaps known fitting events and is diagnostic only; no masked test identifier enters any training window.',
                         'This closed-case track uses known target IDs as supervision and is separate from the verified-history adaptation budget.',
                         'Thirty epochs are fixed before native evaluation; final epoch is evaluated without test-based checkpoint selection.']}
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
    report['status'] = 'completed_here_native_confirmation'
    (artifacts / 'predictions.json').write_text(json.dumps({'predicted': predicted, 'probabilities': probabilities}) + '\n')
    report['predictions_sha256'] = sha256(artifacts / 'predictions.json')
    save_graph(repaired, artifacts / 'repaired.json.gz')
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'name': name, 'result': report['heldout_missing_id'], 'seconds': report['seconds']}), flush=True)

