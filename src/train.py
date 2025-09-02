# Step 7: Define Training and Evaluation Functions
def calculate_metrics(labels, preds):
    labels_np = labels.cpu().numpy()
    preds_np = preds.cpu().numpy()
    precision = precision_score(labels_np, preds_np, average='weighted') * 100
    recall = recall_score(labels_np, preds_np, average='weighted') * 100
    f1 = f1_score(labels_np, preds_np, average='weighted') * 100
    return precision, recall, f1

def train(model, dataloader, optimizer, criterion, device):
    model.train()  # Set the model to training mode
    total_loss = 0
    correct = 0
    total = 0

    for inputs, labels in tqdm(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)  # Move to the correct device
        
        # Ensure labels are in the right format (if needed)
        if len(labels.shape) != 1:  # Check if labels are not 1D
            raise ValueError("Labels must be a 1D tensor.")
        
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update parameters

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)  # Move to the correct device
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(predicted.cpu().numpy())  # Store predictions
            all_labels.extend(labels.cpu().numpy())  # Store true labels

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    precision, recall, f1 = calculate_metrics(torch.tensor(all_labels), torch.tensor(all_preds))
    return avg_loss, accuracy, precision, recall, f1


# Step 8: Train the Model
num_epochs = 10
best_val_acc = 0.0

# Lists to store the metrics for plotting
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

for epoch in range(num_epochs):
    # Train the model and get metrics
    train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
    
    # Evaluate the model on the validation set
    val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate(model, val_loader, criterion, device)

    # Step the scheduler
    scheduler.step()

    # Log the metrics for the current epoch
    print(f'Epoch [{epoch + 1}/{num_epochs}], '
          f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
          f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, '
          f'Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}')
    
    # Optionally log the learning rate
    for param_group in optimizer.param_groups:
        print(f"Epoch {epoch + 1}: Learning rate = {param_group['lr']:.6f}")

    # Save the model if validation accuracy improves
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'epoch': epoch + 1,  # Save the epoch in the checkpoint
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_acc': best_val_acc
        }, 'best_model.pth')