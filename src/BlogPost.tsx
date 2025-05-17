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
      <p  className={date}>{new Date(date).toDateString()}</p>
      <ReactMarkdown remarkPlugins={[remarkGfm]}>
        {content}
      </ReactMarkdown>
    </article>
  );
}
