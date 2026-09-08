import torch.optim as optim

from actions import train, test, get_validation_threshold


def benchmark_epochs(teacher, student, train_loader, val_loader, test_loader, max_epochs, learning_rate, T, device, reverse_distillation=False, n_benchmarks=5):
    delta_epochs = max_epochs//n_benchmarks
    
    student_to_train = student
    optimizer = optim.Adam(student.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    current_total_epochs = 0
    epoch_history = []
    cumulative_train_time = 0.0

    print(f"\n--- Benchmark Epoche (da 0 a {max_epochs} epoche in steps di {delta_epochs}) ---")
    for i in range(n_benchmarks):
        current_epoch = (i+1)*delta_epochs

        trained_student, delta_time = train(teacher=teacher, 
                            student=student_to_train, 
                            train_loader=train_loader, 
                            epochs=delta_epochs,
                            learning_rate=learning_rate, 
                            T=T, 
                            device=device,
                            reverse_distillation=reverse_distillation,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            quiet=True)

        cumulative_train_time += delta_time

        threshold = get_validation_threshold(teacher=teacher,
                                    student=trained_student,
                                    val_loader=val_loader,
                                    device=device,
                                    reverse_distillation=reverse_distillation,
                                    quiet=True)

        metrics, _, _ = test(teacher=teacher, 
                                    student=trained_student, 
                                    test_loader=test_loader, 
                                    device=device,
                                    threshold=threshold,
                                    reverse_distillation=reverse_distillation,
                                    save_vis=False,
                                    quiet=True)

        record = {
            "epoch": current_epoch,
            "cumulative_train_sec": round(cumulative_train_time, 2),
            "threshold": threshold,
            **metrics
        }
        epoch_history.append(record)
        print(f"[CHECKPOINT] Epoch {current_epoch}/{max_epochs} | "
                f"Img AUC: {record['image_roc_auc']:.2f}% | "
                f"Pixel AUC: {record['pixel_roc_auc']:.2f}% | "
                f"F1: {record['f1_score']:.2f}% | "
                f"Time: {record['cumulative_train_sec']}s")

        student_to_train = trained_student

    best_epoch_record = max(epoch_history, key=lambda x: (x["image_roc_auc"], x["f1_score"]))
    return best_epoch_record, epoch_history