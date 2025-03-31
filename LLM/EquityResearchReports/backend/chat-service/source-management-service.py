#Create a simple CRUD service to store and retrieve the source information.
#The service should have the following endpoints:
#1. POST /source: Create a new source
#2. GET /source: Get all sources
#3. GET /source/{id}: Get a source by id
#4. PUT /source/{id}: Update a source by id
#5. DELETE /source/{id}: Delete a source by id
#The source should have the following fields:
#- name: The name of the source
#- pdf: The pdf file of the source
#- created_at: The date the source was created
#- updated_at: The date the source was last updated
#The service should store the source information in a SQLite database.
#Use Flask to create the service.
#Use SQLAlchemy to interact with the database.
#Use Marshmallow to serialize and deserialize the source information.
#Use the following dependencies:
#- Flask
#- SQLAlchemy
#- Marshmallow
#- Flask-SQLAlchemy
#- Flask-Marshmallow
#- SQLite

from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, DateTime
from marshmallow_sqlalchemy import SQLAlchemySchema
import os
import datetime
from flask_cors import CORS
from sqlalchemy import Enum

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{os.path.abspath('sources.db')}"
db = SQLAlchemy(app)
ma = Marshmallow(app)

Base = declarative_base()

class Source(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    company_name = db.Column(db.String(100))
    pdf = db.Column(db.String(100))
    status = db.Column(Enum('active', 'inactive', 'archived', name='status_enum'), nullable=False, default='inactive')
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

class SourceSchema(ma.SQLAlchemyAutoSchema):
    class Meta:
        model = Source

source_schema = SourceSchema()
sources_schema = SourceSchema(many=True)

@app.route('/source', methods=['POST'])
def create_source():
    name = request.json['company_name']
    pdf = request.json['pdf']
    status = request.json.get('status', 'inactive')

    new_source = Source(company_name=name, pdf=pdf, status=status)

    db.session.add(new_source)
    db.session.commit()

    return source_schema.jsonify(new_source)

@app.route('/source/active', methods=['GET'])
def get_active_sources():
    active_sources = Source.query.filter_by(status='active').all()
    result = sources_schema.dump(active_sources)
    return jsonify(result)

@app.route('/source', methods=['GET'])
def get_sources():
    all_sources = Source.query.all()
    result = sources_schema.dump(all_sources)
    return jsonify(result)

@app.route('/source/<id>', methods=['GET'])
def get_source(id):
    source = Source.query.get(id)
    return source_schema.jsonify(source)

@app.route('/source/<id>', methods=['PUT'])
def update_source(id):
    source = Source.query.get(id)
    if not source:
            return jsonify({"error": "Source not found"}), 404

    data = request.json
    source.company_name = data.get('company_name', source.company_name)
    source.pdf = data.get('pdf', source.pdf)
    source.status = data.get('status', source.status)

    db.session.commit()  # Persist the changes to the database

    return source_schema.jsonify(source)

@app.route('/source/<id>', methods=['DELETE'])
def delete_source(id):
    source = Source.query.get(id)

    db.session.delete(source)
    db.session.commit()

    return source_schema.jsonify(source)

if __name__ == '__main__':
    with app.app_context():
        try:
            print("Creating tables...")
            db.create_all()  # Create all tables in the database
            print("Tables created.")
        except Exception as e:
            print(f"Error creating tables: {e}")
    app.run(host='0.0.0.0', port=5003, debug=True)

