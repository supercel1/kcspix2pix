# Generated by Django 2.1 on 2019-09-09 12:21

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('oekaki', '0002_auto_20190907_2230'),
    ]

    operations = [
        migrations.CreateModel(
            name='FakeImage',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('fake_image', models.ImageField(upload_to='pix2pix/fake')),
                ('created_at', models.DateTimeField(auto_now=True)),
            ],
        ),
        migrations.AlterField(
            model_name='files',
            name='files',
            field=models.FileField(upload_to='cyclegan/', verbose_name='ファイル'),
        ),
        migrations.AlterField(
            model_name='images',
            name='image',
            field=models.ImageField(upload_to='pix2pix/'),
        ),
    ]
