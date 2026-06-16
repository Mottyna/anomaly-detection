

def train(teacher, student, train_loader, epochs, learning_rate, T, device):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)

    teacher.eval()  # teacher non viene modifiato
    student.train() # student deve apprendere

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, _ in train_loader:
            inputs = inputs.to(device)
            optimizer.zero_grad()

            # forward pass con il modello teacher, non salvo gradiente
            with torch.no_grad():
                teacher_features = teacher.feature_extractor(inputs) 

            # forward pass con student
            student_logits = student(inputs)

            student_features = student.feature_extractor(inputs)

            loss = criterion(student_features, teacher_features)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")
