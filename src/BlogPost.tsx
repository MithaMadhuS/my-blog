import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import './BlogPost.css'; // optional CSS module

type BlogPostProps = {
  content: string;
  title: string;
  date: string;
};

export default function BlogPost({ content, title, date }: BlogPostProps) {
  return (
    <article className="article">
      <h1>{title}</h1>
      <div className="author-info">
        <img className='profile-picture' src="/my-blog/assets/ganesh.jpeg" alt='Ganesh'></img>
        <span className="author-name">Ganesh</span>
        <span  className='time'>20 min read</span>
        <span className='date'>{new Date(date).toDateString().split(' ').slice(1).join(' ')}</span>
      </div>
      <ReactMarkdown remarkPlugins={[remarkGfm]}>
        {content}
      </ReactMarkdown>
    </article>
  );
}
