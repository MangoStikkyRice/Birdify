from django.db import models

class TrainingMetrics(models.Model):
    date = models.DateTimeField(auto_now_add=True)
    training_loss = models.FloatField()
    validation_loss = models.FloatField()
    accuracy = models.FloatField()
    classification_report = models.TextField()

    def __str__(self):
        return f"Metrics on {self.date}: Acc {self.accuracy:.2f}"
