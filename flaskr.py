#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Flaskr + Babel + Tensorflow
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import os
from sqlite3 import dbapi2 as sqlite3
from flask import Flask, request, session, g, redirect, url_for, abort, \
     render_template, flash, jsonify
from flask_babel import Babel

from tf import image

# Create our little application :)
app = Flask(__name__)
# Read config
app.config.from_pyfile('config.py')

# Hook Babel to our app
babel = Babel(app)

# Check the Accept-Language header and make a smart choice
@babel.localeselector
def get_locale():
    return request.accept_languages.best_match(app.config['LANGUAGES'].keys())


def connect_db():
    """Connects to the specific database."""
    rv = sqlite3.connect(app.config['DATABASE'])
    rv.row_factory = sqlite3.Row
    return rv


def get_db():
    """Opens a new database connection if there is none yet for the
    current application context.
    """
    if not hasattr(g, 'sqlite_db'):
        g.sqlite_db = connect_db()
    return g.sqlite_db


@app.teardown_appcontext
def close_db(error):
    """Closes the database again at the end of the request."""
    print(error)
    if hasattr(g, 'sqlite_db'):
        g.sqlite_db.close()


@app.route('/', methods=['GET'])
def show_entries():
    db = get_db()
    cur = db.execute('select title, text from entries order by id desc')
    entries = cur.fetchall()
    return render_template('show_entries.html', entries=entries)


@app.route('/add', methods=['POST'])
def add_entry():
    if not session.get('logged_in'):
        abort(401)
    db = get_db()
    db.execute('insert into entries (title, text) values (?, ?)',
               [request.form['title'], request.form['text']])
    db.commit()
    flash(('New entry was successfully posted'))
    return redirect(url_for('show_entries'))

@app.route('/guess/<imageUrl>', methods=['GET'])
def guess(imageUrl):
    """
    """
    image.main(imageUrl)
    return jsonify({
        "imageUrl": imageUrl
    })


@app.route('/login', methods=['GET', 'POST'])
def login():
    """
    """
    error = None
    if request.method == 'POST':
        if request.form['username'] != app.config['USERNAME']:
            error = ('Invalid username')
        elif request.form['password'] != app.config['PASSWORD']:
            error = ('Invalid password')
        else:
            session['logged_in'] = True
            flash(('You were logged in'))
            return redirect(url_for('show_entries'))
    return render_template('login.html', error=error)


@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    flash(('You were logged out'))
    return redirect(url_for('show_entries'))

app.run()