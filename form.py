from flask_wtf import FlaskForm 
from wtforms import StringField , SubmitField , IntegerField , FloatField
from wtforms.validators import DataRequired,Length

class submit_data_fish(FlaskForm):
    submit_btn = SubmitField('บันทึก')
    num_fishInput = IntegerField('num_fish',validators=[DataRequired()])
    width_fishInput = FloatField('widht_fish',validators=[DataRequired()])
    height_fishInput = FloatField('height_fish',validators=[DataRequired()])

