"use client";
import React, { useEffect, useState } from 'react';
import { useParams } from 'next/navigation';

interface RaceData {
    name?: string;
    description?: string;
    // add other fields as needed
}

export default function RacePage() {
    const params = useParams();
    const raceId = params?.raceId as string;
    const [data, setData] = useState<RaceData | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        if (!raceId) return;
        async function fetchRace() {
            try {
                const res = await fetch(`/api/race/${raceId}`);
                if (!res.ok) {
                    const err = await res.json();
                    throw new Error(err.error || 'Failed to fetch race');
                }
                const json = await res.json();
                setData(json);
            } catch (e: any) {
                setError(e.message);
            } finally {
                setLoading(false);
            }
        }
        fetchRace();
    }, [raceId]);

    if (loading) return <p className="text-sm text-gray-500">Loading race info…</p>;
    if (error) return <p className="text-sm text-red-500">{error}</p>;
    if (!data) return null;

    return (
        <div className="p-6 max-w-3xl mx-auto">
            <h1 className="text-2xl font-bold mb-4">{data.name ?? 'Race'}</h1>
            {data.description && <p className="mb-4">{data.description}</p>}
            {/* Add more fields as needed */}
        </div>
    );
}
